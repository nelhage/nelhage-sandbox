import z3
from greenery import Pattern


def z3_of_pat(pat, ctx=None):
    concs = [z3_of_conc(c, ctx) for c in pat.concs]
    if len(concs) == 1:
        return concs[0]
    return z3.Union(*concs)


def z3_of_conc(conc, ctx=None):
    mults = [z3_of_mult(m, ctx) for m in conc.mults]
    if len(mults) == 1:
        return mults[0]
    return z3.Concat(*mults)


def z3_of_cc(cclass, ctx=None):
    ranges = []
    for l, r in cclass.ord_ranges:
        if l == r:
            ranges.append(z3.Re(chr(l), ctx))
        else:
            ranges.append(z3.Range(chr(l), chr(r), ctx))

    if len(ranges) == 0 and cclass.negated:
        return z3.AllChar(z3.ReSort(z3.StringSort(ctx)))

    if len(ranges) == 1 and not cclass.negated:
        return ranges[0]

    out = z3.Union(*ranges)
    if cclass.negated:
        out = z3.Complement(out)
    return out


def z3_of_mult(mult, ctx=None):
    if isinstance(mult.multiplicand, Pattern):
        pat = z3_of_pat(mult.multiplicand, ctx)
    else:
        pat = z3_of_cc(mult.multiplicand, ctx)

    multiplier = mult.multiplier
    reps = (multiplier.min.v, multiplier.max.v)
    if reps == (1, 1):
        return pat
    if reps == (0, None):
        return z3.Star(pat)
    if reps == (1, None):
        return z3.Plus(pat)
    if reps == (0, 0):
        return z3.Empty(z3.ReSort(z3.StringSort(ctx)))
    return z3.Loop(pat, reps[0], 0 if reps[1] is None else reps[1])
