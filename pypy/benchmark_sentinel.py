"""Benchmark sentinel_sum across interpreters: sum 10M ints, skipping a -1
sentinel that's placed at ~10% of positions at random.

    python3 benchmark_sentinel.py
    pypy3   benchmark_sentinel.py
"""

import platform
import random
import time

from sums import SENTINEL, sentinel_sum

N = 10_000_000
REPEATS = 5
SENTINEL_FRACTION = 0.10
SEED = 1234


def build_data(n):
    rng = random.Random(SEED)
    data = list(range(n))
    for i in range(n):
        if rng.random() < SENTINEL_FRACTION:
            data[i] = SENTINEL
    return data


def main():
    data = build_data(N)
    n_skip = sum(1 for v in data if v == SENTINEL)

    # Warm-up: lets PyPy's JIT compile the loop before we start timing.
    sentinel_sum(data)

    best = float("inf")
    result = None
    for _ in range(REPEATS):
        start = time.perf_counter()
        result = sentinel_sum(data)
        elapsed = time.perf_counter() - start
        best = min(best, elapsed)

    impl = platform.python_implementation()
    version = platform.python_version()
    rate = N / best

    print(f"{impl} {version}")
    print(f"  list length : {N:,}")
    print(f"  sentinels   : {n_skip:,} ({100*n_skip/N:.1f}%)")
    print(f"  repeats     : {REPEATS}")
    print(f"  best time   : {best:.4f} s")
    print(f"  throughput  : {rate/1e6:.1f} M elements/s")
    print(f"  (sum check) : {result}")


if __name__ == "__main__":
    main()
