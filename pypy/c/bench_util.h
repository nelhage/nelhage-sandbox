#ifndef BENCH_UTIL_H
#define BENCH_UTIL_H

#include <stdint.h>
#include <time.h>

/* mulberry32 PRNG, matching bench_util.js, so the C data is reproducible
   (and built the same way as the JS benchmarks). */
typedef struct {
    uint32_t state;
} rng_t;

static inline void rng_seed(rng_t *r, uint32_t seed) { r->state = seed; }

static inline uint32_t rng_u32(rng_t *r) {
    uint32_t z = (r->state += 0x6d2b79f5u);
    z = (z ^ (z >> 15)) * (z | 1u);
    z ^= z + (z ^ (z >> 7)) * (z | 61u);
    return z ^ (z >> 14);
}

/* float in [0, 1) */
static inline double rng_double(rng_t *r) { return rng_u32(r) / 4294967296.0; }

static inline double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

#endif /* BENCH_UTIL_H */
