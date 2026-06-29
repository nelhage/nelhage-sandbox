"""Benchmark polysum with a MONOMORPHIC call site, for comparison with
benchmark_poly.py.

Identical loop and element count, but every summand is the same type
(AddInt), so the `s.add(total)` call site sees exactly one target. The JIT
can inline it behind a single class guard that never fails -- isolating the
cost that the random 3-type mix adds in benchmark_poly.py.

    python3 benchmark_mono.py
    pypy3 benchmark_mono.py
"""

import platform
import random
import time

from summands import AddInt, polysum

N = 5_000_000  # number of summands (matches benchmark_poly.py)
REPEATS = 5  # timed repetitions; we report the best one
SEED = 1234


def build_data(n):
    rng = random.Random(SEED)
    # Single type, random small int value for each summand.
    return [AddInt(rng.randint(0, 9)) for _ in range(n)]


def main():
    data = build_data(N)

    # Warm-up: lets PyPy's JIT compile the loop before we start timing.
    polysum(data)

    best = float("inf")
    result = None
    for _ in range(REPEATS):
        start = time.perf_counter()
        result = polysum(data)
        elapsed = time.perf_counter() - start
        best = min(best, elapsed)

    impl = platform.python_implementation()
    version = platform.python_version()
    rate = N / best

    print(f"{impl} {version}")
    print(f"  summands    : {N:,} (monomorphic: all AddInt)")
    print(f"  repeats     : {REPEATS}")
    print(f"  best time   : {best:.4f} s")
    print(f"  throughput  : {rate/1e6:.1f} M calls/s")
    print(f"  (sum check) : {result}")


if __name__ == "__main__":
    main()
