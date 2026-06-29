#include "sums.h"

long sum_ints(const int *values, size_t n) {
    long total = 0;
    for (size_t i = 0; i < n; i++) {
        total += values[i];
    }
    return total;
}

long sentinel_sum(const int *values, size_t n) {
    /* Skip the -1 sentinel; sum everything else. */
    long total = 0;
    for (size_t i = 0; i < n; i++) {
        int v = values[i];
        if (v == SENTINEL) continue;
        total += v;
    }
    return total;
}
