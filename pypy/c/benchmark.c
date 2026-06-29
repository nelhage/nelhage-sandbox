/* Runs the four headline benchmarks in C, mirroring the Python/JS versions.
   Build with `make` (see Makefile) and run ./benchmark. */
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include "bench_util.h"
#include "sum_floats.h"
#include "summands.h"

#define SEED 1234u
#define REPEATS 5
#define N_FLOATS 10000000UL
#define N_SUMMANDS 5000000UL
#define NAN_FRACTION 0.10

static double *build_floats(size_t n) {
    double *data = malloc(n * sizeof *data);
    for (size_t i = 0; i < n; i++) data[i] = (double)i * 0.5;
    return data;
}

static double *build_nansum(size_t n, double frac) {
    rng_t rng;
    rng_seed(&rng, SEED);
    double *data = build_floats(n);
    for (size_t i = 0; i < n; i++) {
        if (rng_double(&rng) < frac) data[i] = NAN;
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
                   double check) {
    printf("  %-30s best %.4f s  %8.1f %s  (check %.0f)\n", label, best,
           n / best / 1e6, unit, check);
}

int main(void) {
    double *floats = build_floats(N_FLOATS);
    double *nans = build_nansum(N_FLOATS, NAN_FRACTION);
    summand_base **mono = build_mono(N_SUMMANDS);
    summand_base **poly = build_poly(N_SUMMANDS);

    double best, check;
    int icheck;

    printf("C (gcc, see Makefile)  N_floats=%lu N_summands=%lu, %d reps best\n",
           N_FLOATS, N_SUMMANDS, REPEATS);

    best = 1e30;
    check = 0;
    for (int r = 0; r < REPEATS; r++) {
        double s = now_sec();
        check = sum_floats(floats, N_FLOATS);
        double e = now_sec() - s;
        if (e < best) best = e;
    }
    report("plain sum (10M floats)", N_FLOATS, "M elem/s", best, check);

    best = 1e30;
    check = 0;
    for (int r = 0; r < REPEATS; r++) {
        double s = now_sec();
        check = nansum_floats(nans, N_FLOATS);
        double e = now_sec() - s;
        if (e < best) best = e;
    }
    report("nansum, 10% random NaN", N_FLOATS, "M elem/s", best, check);

    best = 1e30;
    icheck = 0;
    for (int r = 0; r < REPEATS; r++) {
        double s = now_sec();
        icheck = polysum(mono, N_SUMMANDS);
        double e = now_sec() - s;
        if (e < best) best = e;
    }
    report("polysum, monomorphic", N_SUMMANDS, "M calls/s", best, icheck);

    best = 1e30;
    icheck = 0;
    for (int r = 0; r < REPEATS; r++) {
        double s = now_sec();
        icheck = polysum(poly, N_SUMMANDS);
        double e = now_sec() - s;
        if (e < best) best = e;
    }
    report("polysum, polymorphic (3-way)", N_SUMMANDS, "M calls/s", best, icheck);

    return 0;
}
