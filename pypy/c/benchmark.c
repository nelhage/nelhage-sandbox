/* Runs the four headline benchmarks in C, mirroring the Python/JS versions.
   Build with `make` (see Makefile) and run ./benchmark. */
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include "bench_util.h"
#include "summands.h"
#include "sums.h"

#define SEED 1234u
#define REPEATS 5
#define N_INTS 10000000UL
#define N_SUMMANDS 5000000UL
#define SENTINEL_FRACTION 0.10

static int *build_ints(size_t n) {
    int *data = malloc(n * sizeof *data);
    for (size_t i = 0; i < n; i++) data[i] = (int)i;
    return data;
}

static int *build_sentinel(size_t n, double frac) {
    rng_t rng;
    rng_seed(&rng, SEED);
    int *data = build_ints(n);
    for (size_t i = 0; i < n; i++) {
        if (rng_double(&rng) < frac) data[i] = SENTINEL;
    }
    return data;
}

static summand_base **build_poly(size_t n) {
    rng_t rng;
    rng_seed(&rng, SEED);
    summand_base **data = malloc(n * sizeof *data);
    for (size_t i = 0; i < n; i++) {
        summand_ctor *kind = SUMMAND_KINDS[(int)(rng_double(&rng) * 3.0)];
        data[i] = kind((int)(rng_double(&rng) * 10.0));
    }
    return data;
}

static summand_base **build_mono(size_t n) {
    rng_t rng;
    rng_seed(&rng, SEED);
    summand_base **data = malloc(n * sizeof *data);
    for (size_t i = 0; i < n; i++) {
        data[i] = make_add_int((int)(rng_double(&rng) * 10.0));
    }
    return data;
}

static void report(const char *label, size_t n, const char *unit, double best,
                   long check) {
    printf("  %-30s best %.4f s  %8.1f %s  (check %ld)\n", label, best,
           n / best / 1e6, unit, check);
}

int main(void) {
    int *ints = build_ints(N_INTS);
    int *sentinel = build_sentinel(N_INTS, SENTINEL_FRACTION);
    summand_base **mono = build_mono(N_SUMMANDS);
    summand_base **poly = build_poly(N_SUMMANDS);

    double best;
    long check;

    printf("C (gcc, see Makefile)  N_ints=%lu N_summands=%lu, %d reps best\n",
           N_INTS, N_SUMMANDS, REPEATS);

    best = 1e30;
    check = 0;
    for (int r = 0; r < REPEATS; r++) {
        double s = now_sec();
        check = sum_ints(ints, N_INTS);
        double e = now_sec() - s;
        if (e < best) best = e;
    }
    report("plain sum (10M ints)", N_INTS, "M elem/s", best, check);

    best = 1e30;
    check = 0;
    for (int r = 0; r < REPEATS; r++) {
        double s = now_sec();
        check = sentinel_sum(sentinel, N_INTS);
        double e = now_sec() - s;
        if (e < best) best = e;
    }
    report("sentinel sum, 10% skipped", N_INTS, "M elem/s", best, check);

    best = 1e30;
    check = 0;
    for (int r = 0; r < REPEATS; r++) {
        double s = now_sec();
        check = polysum(mono, N_SUMMANDS);
        double e = now_sec() - s;
        if (e < best) best = e;
    }
    report("polysum, monomorphic", N_SUMMANDS, "M calls/s", best, check);

    best = 1e30;
    check = 0;
    for (int r = 0; r < REPEATS; r++) {
        double s = now_sec();
        check = polysum(poly, N_SUMMANDS);
        double e = now_sec() - s;
        if (e < best) best = e;
    }
    report("polysum, polymorphic (3-way)", N_SUMMANDS, "M calls/s", best, check);

    return 0;
}
