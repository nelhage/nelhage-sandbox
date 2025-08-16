import time
from dataclasses import dataclass
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import nn, random
from jax.example_libraries import optimizers
from jax.nn import initializers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


def init_network_params(layer_sizes, key, init):
    """Initialize network parameters"""
    keys = random.split(key, len(layer_sizes))
    params = []

    for i in range(len(layer_sizes) - 1):
        w_key = keys[i]
        n_in, n_out = layer_sizes[i], layer_sizes[i + 1]

        W = init(w_key, (n_in, n_out), jnp.float32)
        b = jnp.zeros(n_out)
        params.append((W, b))

    return params


def relu(x):
    """ReLU activation function."""
    return jnp.maximum(0, x)


@partial(jax.vmap, in_axes=(None, 0))
def forward_pass(params, x):
    """Forward pass through the network."""
    # Flatten input if needed
    x = x.reshape(-1)  # Flatten 28x28 to 784

    # Forward through hidden layers with ReLU
    for i, (W, b) in enumerate(params[:-1]):
        x = relu(jnp.dot(x, W) + b)

    # Output layer (no activation)
    W, b = params[-1]
    x = jnp.dot(x, W) + b

    return x


def cross_entropy_loss(outputs, y):
    """Cross-entropy loss function."""
    logprobs = nn.log_softmax(outputs, axis=1)
    return -jnp.sum(y * logprobs, axis=1)


def accuracy(outputs, y):
    """Calculate accuracy on a batch."""

    predicted_classes = jnp.argmax(outputs, axis=1)
    true_classes = jnp.argmax(y, axis=1)
    return jnp.mean(predicted_classes == true_classes)


def forward_and_accuracy(params, x_batch, y_batch):
    out = forward_pass(params, x_batch)
    per_seq_accuracy = accuracy(out, y_batch)
    return jnp.mean(per_seq_accuracy)


def train_step(
    step, params, x_batch, y_batch, opt_state, loss_fn, opt_update, get_params
):
    """Single training step."""

    def fwd_and_loss(params):
        out = forward_pass(params, x_batch)
        loss = loss_fn(out, y_batch)
        return jnp.mean(loss)

    val_and_grad = jax.value_and_grad(fwd_and_loss)

    loss_val, grads = val_and_grad(params)

    opt_state = opt_update(step, grads, opt_state)
    params = get_params(opt_state)
    return params, opt_state, loss_val


def load_and_preprocess_data(zero_center: bool = False):
    """Load and preprocess MNIST data."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize pixel values to [0, 1]
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    # Convert labels to one-hot encoding
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    if zero_center:
        x_train = 2 * (x_train - 0.5)
        x_test = 2 * (x_test - 0.5)

    return (x_train, y_train), (x_test, y_test)


def create_batches(x, y, batch_size):
    """Create mini-batches from data."""
    n_samples = x.shape[0]
    indices = np.random.permutation(n_samples)

    for i in range(0, n_samples, batch_size):
        batch_indices = indices[i : i + batch_size]
        yield x[batch_indices], y[batch_indices]


@dataclass
class Config:
    epochs: int = 5
    batch_size: int = 128
    learning_rate: float = 1e-3
    init: initializers.Initializer = initializers.variance_scaling(
        mode="fan_in", scale=2, distribution="normal"
    )

    loss_fn: Callable = cross_entropy_loss
    random_seed: jax.typing.ArrayLike = 42
    measure_accuracy: bool = True


def train_loop(cfg: Config, params, x_train, y_train, x_test=None, y_test=None):
    opt_init, opt_update, get_params = optimizers.adam(cfg.learning_rate)
    opt_state = jax.vmap(opt_init)(params)

    train_one_step = partial(
        train_step,
        loss_fn=cfg.loss_fn,
        opt_update=opt_update,
        get_params=get_params,
    )
    train_jit = jax.jit(jax.vmap(train_one_step, in_axes=(None, 0, None, None, 0)))
    batch_accuracy = jax.jit(jax.vmap(forward_and_accuracy, in_axes=(0, None, None)))

    n_data = x_train.shape[0]
    (n_model,) = set(p.shape[0] for p in jax.tree.leaves(params))
    n_ex = n_data * n_model

    step = jnp.array(1)
    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        n_batches = 0

        # Shuffle and batch data
        start_epoch = time.perf_counter()
        for x_batch, y_batch in create_batches(x_train, y_train, cfg.batch_size):
            params, opt_state, batch_loss = train_jit(
                step,
                params,
                x_batch,
                y_batch,
                opt_state,
            )
            epoch_loss += batch_loss
            n_batches += 1
            step = step + 1
        end_epoch = time.perf_counter()

        t_epoch = end_epoch - start_epoch

        # Calculate average loss and accuracy
        avg_loss = epoch_loss / n_batches

        if cfg.measure_accuracy:
            train_acc = jax.device_get(
                batch_accuracy(params, x_train[:1000], y_train[:1000])
            )
        else:
            train_acc = np.zeros_like(avg_loss)

        if x_test is not None and y_test is not None:
            test_acc = jax.device_get(batch_accuracy(params, x_test, y_test))
        else:
            test_acc = np.zeros_like(train_acc)

        with np.printoptions(precision=3, floatmode="fixed"):
            print(
                f"Epoch {epoch + 1:2d}: Loss = {avg_loss}, "
                + (
                    f"Train Acc = {train_acc}, Test Acc = {test_acc:}  "
                    if cfg.measure_accuracy
                    else ""
                )
                + f"t={t_epoch:.2f}s ex/sec={n_ex / t_epoch:.1f}"
            )
    return params


def train_one(
    *,
    cfg: Config = Config(),
):
    """Main training loop."""
    key = random.PRNGKey(cfg.random_seed)

    # Load data
    print("Loading MNIST data...")
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()

    # Network architecture: 784 -> 256 -> 256 -> 10
    layer_sizes = [28 * 28, 256, 256, 10]

    # Initialize network parameters
    print("Initializing network...")
    params = init_network_params(layer_sizes, key, init=cfg.init)

    # Initialize optimizer (Adam)
    opt_init, opt_update, get_params = optimizers.adam(cfg.learning_rate)
    opt_state = opt_init(params)

    # Training parameters

    print("Starting training...")
    print(f"Network architecture: {' -> '.join(map(str, layer_sizes))}")
    print(f"Learning rate: {cfg.learning_rate}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Epochs: {cfg.epochs}")
    print("-" * 50)

    params = jax.tree.map(lambda p: p[None, ...], params)
    params = train_loop(cfg, params, x_train, y_train, x_test, y_test)
    params = jax.tree.map(lambda p: jnp.squeeze(p, 0), params)

    return params


# def train_step(step, params, x_batch, y_batch, opt_state, opt_update, get_params):
