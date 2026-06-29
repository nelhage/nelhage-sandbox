"""Benchmark harness for sum_ints across interpreters.

Run under whichever interpreter is invoked, e.g.:
    python3 benchmark.py
    pypy3 benchmark.py

Builds a fixed list of ints and times the pure-Python summing loop over
several repetitions, reporting the best (fastest) time as throughput.
"""

import platform
import time

from sums import sum_ints

N = 10_000_000  # length of the list to sum
REPEATS = 5  # timed repetitions; we report the best one


def build_data(n):
    return list(range(n))


def main():
    data = build_data(N)

    # Warm-up: lets PyPy's JIT compile the loop before we start timing.
    sum_ints(data)

    best = float("inf")
    result = None
    for _ in range(REPEATS):
        start = time.perf_counter()
        result = sum_ints(data)
        elapsed = time.perf_counter() - start
        best = min(best, elapsed)

    impl = platform.python_implementation()
    version = platform.python_version()
    rate = N / best

    print(f"{impl} {version}")
    print(f"  list length : {N:,}")
    print(f"  repeats     : {REPEATS}")
    print(f"  best time   : {best:.4f} s")
    print(f"  throughput  : {rate/1e6:.1f} M elements/s")
    print(f"  (sum check) : {result}")


if __name__ == "__main__":
    main()
