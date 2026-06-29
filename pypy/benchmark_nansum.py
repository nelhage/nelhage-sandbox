"""Benchmark harness for nansum_floats across interpreters.

Same shape as benchmark.py, but the data has ~10% of its elements replaced
with NaN (at random, fixed seed) and the loop skips NaNs.

    python3 benchmark_nansum.py
    pypy3 benchmark_nansum.py
"""

import math
import platform
import random
import time

from sum_floats import nansum_floats

N = 10_000_000  # length of the list to sum
REPEATS = 5  # timed repetitions; we report the best one
NAN_FRACTION = 0.10
SEED = 1234


def build_data(n):
    rng = random.Random(SEED)
    nan = float("nan")
    # Same base values as benchmark.py, with ~NAN_FRACTION randomly set to NaN.
    data = [i * 0.5 for i in range(n)]
    for i in range(n):
        if rng.random() < NAN_FRACTION:
            data[i] = nan
    return data


def main():
    data = build_data(N)
    n_nan = sum(1 for v in data if math.isnan(v))

    # Warm-up: lets PyPy's JIT compile the loop before we start timing.
    nansum_floats(data)

    best = float("inf")
    result = None
    for _ in range(REPEATS):
        start = time.perf_counter()
        result = nansum_floats(data)
        elapsed = time.perf_counter() - start
        best = min(best, elapsed)

    impl = platform.python_implementation()
    version = platform.python_version()
    rate = N / best

    print(f"{impl} {version}")
    print(f"  list length : {N:,}")
    print(f"  NaN count   : {n_nan:,} ({100*n_nan/N:.1f}%)")
    print(f"  repeats     : {REPEATS}")
    print(f"  best time   : {best:.4f} s")
    print(f"  throughput  : {rate/1e6:.1f} M elements/s")
    print(f"  (sum check) : {result}")


if __name__ == "__main__":
    main()
