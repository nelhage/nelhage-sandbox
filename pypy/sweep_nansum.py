"""Sweep nansum throughput across a range of p(NaN).

Rebuilds the 10M-element list at each NaN fraction and times nansum_floats,
printing one row per fraction. Run under any interpreter:

    python3 sweep_nansum.py
    pypy3   sweep_nansum.py
"""

import platform
import random
import sys
import time

from sum_floats import nansum_floats

N = 10_000_000
REPEATS = 5
SEED = 1234
DEFAULT_FRACTIONS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
# Pass fractions on the command line to sweep just those (e.g. one per
# process, to avoid the JIT compiling the loop once and reusing that trace
# across every fraction).
FRACTIONS = [float(a) for a in sys.argv[1:]] or DEFAULT_FRACTIONS


def build_data(n, frac):
    rng = random.Random(SEED)
    nan = float("nan")
    data = [i * 0.5 for i in range(n)]
    for i in range(n):
        if rng.random() < frac:
            data[i] = nan
    return data


def best_time(data):
    nansum_floats(data)  # warm-up for this dataset
    best = float("inf")
    for _ in range(REPEATS):
        start = time.perf_counter()
        nansum_floats(data)
        best = min(best, time.perf_counter() - start)
    return best


def main():
    print(f"{platform.python_implementation()} {platform.python_version()}  "
          f"(N={N:,}, {REPEATS} reps, best)")
    print(f"  {'p(NaN)':>7} {'best (s)':>10} {'M elem/s':>10}")
    for frac in FRACTIONS:
        data = build_data(N, frac)
        best = best_time(data)
        print(f"  {frac:>7.2f} {best:>10.4f} {N / best / 1e6:>10.1f}")


if __name__ == "__main__":
    main()
