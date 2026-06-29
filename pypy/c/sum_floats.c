#include "sum_floats.h"

double sum_floats(const double *values, size_t n) {
    double total = 0.0;
    for (size_t i = 0; i < n; i++) {
        total += values[i];
    }
    return total;
}

double nansum_floats(const double *values, size_t n) {
    /* nansum semantics: skip NaN. (x != x) is true only for NaN. */
    double total = 0.0;
    for (size_t i = 0; i < n; i++) {
        double x = values[i];
        if (x != x) continue;
        total += x;
    }
    return total;
}
