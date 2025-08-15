import time
from functools import partial

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


def forward_and_loss(params, x_batch, y_batch):
    """Average loss over a batch."""
    out = forward_pass(params, x_batch)
    per_seq_loss = cross_entropy_loss(out, y_batch)
    return jnp.mean(per_seq_loss)


def forward_and_accuracy(params, x_batch, y_batch):
    out = forward_pass(params, x_batch)
    per_seq_accuracy = accuracy(out, y_batch)
    return jnp.mean(per_seq_accuracy)


def train_step(step, params, x_batch, y_batch, opt_state, opt_update, get_params):
    """Single training step."""
    val_and_grad = jax.value_and_grad(
        lambda params: forward_and_loss(params, x_batch, y_batch)
    )

    loss_val, grads = val_and_grad(params)

    opt_state = opt_update(step, grads, opt_state)
    params = get_params(opt_state)
    return params, opt_state, loss_val


def load_and_preprocess_data():
    """Load and preprocess MNIST data."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize pixel values to [0, 1]
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    # Convert labels to one-hot encoding
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)


def create_batches(x, y, batch_size):
    """Create mini-batches from data."""
    n_samples = x.shape[0]
    indices = np.random.permutation(n_samples)

    for i in range(0, n_samples, batch_size):
        batch_indices = indices[i : i + batch_size]
        yield x[batch_indices], y[batch_indices]


# JIT compile training step
train_step_jit = jax.jit(train_step, static_argnames=("opt_update", "get_params"))
fwd_and_accuracy_jit = jax.jit(forward_and_accuracy)


def train_one(
    *,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-3,
    init=initializers.variance_scaling(mode="fan_in", scale=2, distribution="normal"),
):
    """Main training loop."""
    # Set random seed for reproducibility
    key = random.PRNGKey(42)

    # Load data
    print("Loading MNIST data...")
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()

    # Network architecture: 784 -> 256 -> 256 -> 10
    layer_sizes = [28 * 28, 256, 256, 10]

    # Initialize network parameters
    print("Initializing network...")
    params = init_network_params(layer_sizes, key, init=init)

    # Initialize optimizer (Adam)
    learning_rate = lr
    opt_init, opt_update, get_params = optimizers.adam(learning_rate)
    opt_state = opt_init(params)

    # Training parameters

    print("Starting training...")
    print(f"Network architecture: {' -> '.join(map(str, layer_sizes))}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print("-" * 50)

    # Training loop
    n_ex = x_train.shape[0]

    step = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0

        # Shuffle and batch data
        t_start = time.time()
        for x_batch, y_batch in create_batches(x_train, y_train, batch_size):
            params, opt_state, batch_loss = train_step_jit(
                step,
                params,
                x_batch,
                y_batch,
                opt_state,
                opt_update,
                get_params,
            )
            epoch_loss += batch_loss
            n_batches += 1
            step += 1
        t_end = time.time()
        t_step = t_end - t_start

        # Calculate average loss and accuracy
        avg_loss = epoch_loss / n_batches
        train_acc = fwd_and_accuracy_jit(
            params, x_train[:1000], y_train[:1000]
        )  # Sample for speed
        test_acc = fwd_and_accuracy_jit(params, x_test, y_test)

        print(
            f"Epoch {epoch + 1:2d}: Loss = {avg_loss:.4f}, "
            f"Train Acc = {train_acc:.4f}, Test Acc = {test_acc:.4f} t={t_step:.2f}s ex/s={n_ex / t_step:.1f}"
        )

    print("-" * 50)
    print("Training completed!")

    # Final evaluation
    print(f"Final test accuracy: {test_acc:.4f}")

    return params
