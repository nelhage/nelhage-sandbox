/* Sweep nansum throughput across p(NaN), mirroring sweep_nansum.{py,js}.
   Pass fractions as CLI args, else a default set is used. */
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include "bench_util.h"
#include "sum_floats.h"

#define SEED 1234u
#define REPEATS 5
#define N 10000000UL

static double DEFAULT_FRACTIONS[] = {0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
                                     0.6,  0.7, 0.8, 0.9, 0.95};

static double *build_data(size_t n, double frac) {
    rng_t rng;
    rng_seed(&rng, SEED);
    double *data = malloc(n * sizeof *data);
    for (size_t i = 0; i < n; i++) data[i] = (double)i * 0.5;
    for (size_t i = 0; i < n; i++) {
        if (rng_double(&rng) < frac) data[i] = NAN;
    }
    return data;
}

static double best_time(const double *data, size_t n) {
    double best = 1e30;
    volatile double sink = 0;
    for (int r = 0; r < REPEATS; r++) {
        double s = now_sec();
        sink = nansum_floats(data, n);
        double e = now_sec() - s;
        if (e < best) best = e;
    }
    (void)sink;
    return best;
}

int main(int argc, char **argv) {
    double *fractions;
    size_t nfrac;
    if (argc > 1) {
        nfrac = (size_t)(argc - 1);
        fractions = malloc(nfrac * sizeof *fractions);
        for (size_t i = 0; i < nfrac; i++) fractions[i] = atof(argv[i + 1]);
    } else {
        fractions = DEFAULT_FRACTIONS;
        nfrac = sizeof DEFAULT_FRACTIONS / sizeof DEFAULT_FRACTIONS[0];
    }

    printf("C (gcc)  N=%lu, %d reps, best\n", N, REPEATS);
    printf("  %7s %10s %10s\n", "p(NaN)", "best (s)", "M elem/s");
    for (size_t i = 0; i < nfrac; i++) {
        double *data = build_data(N, fractions[i]);
        double best = best_time(data, N);
        printf("  %7.2f %10.4f %10.1f\n", fractions[i], best, N / best / 1e6);
        free(data);
    }
    return 0;
}
