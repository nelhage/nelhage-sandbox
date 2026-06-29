"""Sweep sentinel_sum throughput across a range of p(sentinel).

Rebuilds the 10M-element list at each sentinel fraction and times sentinel_sum,
printing one row per fraction. Run under any interpreter:

    python3 sweep_sentinel.py
    pypy3   sweep_sentinel.py

Pass fractions on the command line to sweep just those (e.g. one per process,
to avoid the JIT compiling the loop once and reusing that trace).
"""

import platform
import random
import sys
import time

from sums import SENTINEL, sentinel_sum

N = 10_000_000
REPEATS = 5
SEED = 1234
DEFAULT_FRACTIONS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
FRACTIONS = [float(a) for a in sys.argv[1:]] or DEFAULT_FRACTIONS


def build_data(n, frac):
    rng = random.Random(SEED)
    data = list(range(n))
    for i in range(n):
        if rng.random() < frac:
            data[i] = SENTINEL
    return data


def best_time(data):
    sentinel_sum(data)  # warm-up for this dataset
    best = float("inf")
    for _ in range(REPEATS):
        start = time.perf_counter()
        sentinel_sum(data)
        best = min(best, time.perf_counter() - start)
    return best


def main():
    print(f"{platform.python_implementation()} {platform.python_version()}  "
          f"(N={N:,}, {REPEATS} reps, best)")
    print(f"  {'p(sent)':>7} {'best (s)':>10} {'M elem/s':>10}")
    for frac in FRACTIONS:
        data = build_data(N, frac)
        best = best_time(data)
        print(f"  {frac:>7.2f} {best:>10.4f} {N / best / 1e6:>10.1f}")


if __name__ == "__main__":
    main()
