"""Sum a list of floats using an explicit Python loop."""


def sum_floats(values):
    total = 0.0
    for v in values:
        total += v
    return total


def nansum_floats(values):
    # nansum semantics: treat NaN as 0 (skip it). `v != v` is true only for
    # NaN, so it avoids a math.isnan() call in the hot loop.
    total = 0.0
    for v in values:
        if v != v:
            continue
        total += v
    return total
