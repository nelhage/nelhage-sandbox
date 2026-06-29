#ifndef SUMS_H
#define SUMS_H

#include <stddef.h>

#define SENTINEL (-1)

/* Element values fit in int; the running total can reach ~5e13, so it's long. */
long sum_ints(const int *values, size_t n);
long sentinel_sum(const int *values, size_t n);

#endif /* SUMS_H */
