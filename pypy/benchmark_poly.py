"""Benchmark polysum (polymorphic dispatch) across interpreters.

The list holds a random mix of the three Summand subclasses, so the
`s.add(total)` call site sees three target methods -- a megamorphic-ish
call that the JIT can't collapse to a single inlined target.

    python3 benchmark_poly.py
    pypy3 benchmark_poly.py
"""

import platform
import random
import time

from summands import KINDS, polysum

N = 5_000_000  # number of summands
REPEATS = 5  # timed repetitions; we report the best one
SEED = 1234


def build_data(n):
    rng = random.Random(SEED)
    # Random type mix and random small int value for each summand.
    return [rng.choice(KINDS)(rng.randint(0, 9)) for _ in range(n)]


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
    print(f"  summands    : {N:,} (random mix of {len(KINDS)} types)")
    print(f"  repeats     : {REPEATS}")
    print(f"  best time   : {best:.4f} s")
    print(f"  throughput  : {rate/1e6:.1f} M calls/s")
    print(f"  (sum check) : {result}")


if __name__ == "__main__":
    main()
