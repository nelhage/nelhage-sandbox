"""Integer summing loops: a plain sum and a sentinel-skipping sum.

Everything is int, so these are directly comparable with the polysum
(vtable-dispatch) benchmarks, which are also int. The sentinel sum is the int
analogue of the old nansum: instead of skipping NaN it skips a -1 sentinel.
"""

SENTINEL = -1


def sum_ints(values):
    total = 0
    for v in values:
        total += v
    return total


def sentinel_sum(values):
    # Skip the sentinel; sum everything else. The data values are all >= 0,
    # so -1 is an unambiguous "don't add this" marker.
    total = 0
    for v in values:
        if v == SENTINEL:
            continue
        total += v
    return total
