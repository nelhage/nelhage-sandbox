"""Benchmark harness for sum_floats across interpreters.

Run under whichever interpreter is invoked, e.g.:
    python3 benchmark.py
    pypy3 benchmark.py

Builds a fixed list of floats and times the pure-Python summing loop over
several repetitions, reporting the best (fastest) time as throughput.
"""

import platform
import sys
import time

from sum_floats import sum_floats

N = 10_000_000  # length of the list to sum
REPEATS = 5  # timed repetitions; we report the best one


def build_data(n):
    # Deterministic, varied floats so the result isn't trivially constant.
    return [i * 0.5 for i in range(n)]


def main():
    data = build_data(N)

    # Warm-up: lets PyPy's JIT compile the loop before we start timing.
    sum_floats(data)

    best = float("inf")
    result = None
    for _ in range(REPEATS):
        start = time.perf_counter()
        result = sum_floats(data)
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
