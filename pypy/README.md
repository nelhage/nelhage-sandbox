# PyPy vs CPython vs V8: pure-language loop benchmarks

A small harness comparing equivalent summing loops on three engines —
CPython, PyPy, and V8 (Node) — across a few workloads that stress their
JITs differently.

## Environment

All three engines come from the `pypy` devShell in the repo's `flake.nix`
(`pkgs.python3` for CPython, `pkgs.pypy3` for PyPy, `pkgs.nodejs` for V8). The
`.envrc` activates it via direnv, so `python3`, `pypy3`, and `node` are all on
`PATH` inside this directory.

## Files

The Python and JavaScript ports mirror each other one-to-one.

- `sum_floats.{py,js}` — the functions under test: a plain summing loop and an
  nansum (skip-NaN) loop.
- `summands.{py,js}` — a `Summand` base class with three subclasses (`AddInt`,
  `SquareInt`, `Identity`) and `polysum`, which loops calling `s.add(total)`.
- `benchmark.{py,js}` — plain float sum over a 10M-element list.
- `benchmark_nansum.{py,js}` — nansum over 10M floats with ~10% randomly NaN.
- `benchmark_poly.{py,js}` — `polysum` over 5M summands, a **random mix** of
  the three types (polymorphic call site).
- `benchmark_mono.{py,js}` — `polysum` over 5M summands, all one type
  (monomorphic call site), for comparison with the polymorphic case.
- `sweep_nansum.{py,js}` — sweep nansum throughput across a range of p(NaN).
  With no args it sweeps a default set of fractions in one process; pass
  fractions as CLI args (e.g. `sweep_nansum.py 0.5`) to measure one per
  process and avoid the JIT reusing a single early-compiled trace.
- `bench_util.js` — shared JS helpers (seeded PRNG + best-of-N timer).
- `run_benchmark.sh` — runs `benchmark.py` under CPython and PyPy.

Each benchmark times 5 reps and reports the best, after one warm-up call so the
JIT is compiled before timing. Each loop is written in the idiomatic shape for
its language — `for v in values` in Python, an indexed `for (let i ...)` in JS
— rather than a literal translation, so we're comparing the code people
actually write on each engine (see the methodology note below).

## Running

```sh
./run_benchmark.sh                 # plain sum, CPython + PyPy
# or any benchmark under any engine:
python3 benchmark_poly.py
pypy3   benchmark_poly.py
node    benchmark_poly.js
```

## Results (rough, one machine, x86_64)

| Benchmark                      | CPython 3.13 | PyPy 3.11 | V8 13.6 / Node 24 |
|--------------------------------|--------------|-----------|-------------------|
| plain `sum` (10M floats)       | ~85 M/s      | ~1870 M/s | ~1706 M/s         |
| `nansum`, 10% random NaN       | ~52 M/s      | ~536 M/s  | ~275 M/s          |
| `polysum`, monomorphic         | ~33 M/s      | ~334 M/s  | ~618 M/s          |
| `polysum`, polymorphic (3-way) | ~24 M/s      | ~117 M/s  | ~156 M/s          |

(Throughput is M elements/s for the sums, M calls/s for `polysum`. Engines
each use their own seeded data, so the checksums differ across languages but
are stable within an engine.)

## What the numbers show

**Both JITs crush the CPython interpreter (20×+ on the best case). Between the
two, the split is clean: PyPy wins the numeric array loops, V8 wins the object
method dispatch.**

- **Plain float-add** — PyPy and V8 are essentially tied (~1870 vs ~1706 M/s),
  both ~20× CPython. Both compile the hot loop to a bare add over an
  unboxed-double array; at this point it's close to memory-bandwidth bound.
- **nansum** — PyPy stays ~2× ahead of V8 (536 vs 275). The `v != v` test plus
  10% *random* NaNs is an unpredictable branch that hurts both, but PyPy's
  trace + bridge handling absorbs it better than V8 here.
- **Monomorphic dispatch** — V8 pulls ahead (618 vs 334). A single-type call
  site is exactly what V8's hidden-class + inline-cache machinery is built for;
  it inlines the one `add` and the per-call overhead nearly vanishes.
- **Polymorphic dispatch** — V8 stays ahead (156 vs 117) even though it loses
  the most going mono→poly: V8 drops ~4.0× (618→156), PyPy ~2.9× (334→117), and
  CPython only ~1.4× (33→24, it was already paying full dispatch cost). V8 falls
  furthest precisely because its monomorphic case was so heavily inlined, but
  its polymorphic inline caches still leave it on top in absolute terms.

The takeaway: on idiomatic code there's no single winner — PyPy's tracing JIT
is strongest on tight numeric loops, while V8's inline-cache-driven design wins
on method dispatch over objects.

## nansum across p(NaN): a branch-prediction curve

Sweeping the NaN fraction (each point measured in its own process, so the JIT
compiles fresh for that fraction) gives throughput in M elem/s:

| p(NaN) | CPython 3.13 | PyPy 3.11 | V8 13.6 / Node 24 |
|--------|--------------|-----------|-------------------|
| 0.05   | ~56          | ~768      | ~311              |
| 0.10   | ~58          | ~545      | ~275              |
| 0.20   | ~56          | ~345      | ~230              |
| 0.30   | ~53          | ~255      | ~202              |
| 0.40   | ~44          | ~203      | ~178              |
| 0.50   | ~51          | ~181      | ~166              |
| 0.60   | ~54          | ~191      | ~176              |
| 0.70   | ~57          | ~255      | ~201              |
| 0.80   | ~56          | ~345      | ~240              |
| 0.90   | ~63          | ~556      | ~316              |
| 0.95   | ~66          | ~821      | ~385              |

All three trace a **U**: throughput bottoms out near **p = 0.5** and recovers
toward both extremes. That's the signature of **branch (mis)prediction** on the
`if v != v` test — at p ≈ 0.5 the branch is maximally unpredictable (~50%
mispredict rate), while at p → 0 or p → 1 it's almost always the same way and
the CPU predicts it for free.

- **CPython** is nearly flat (~44–66 M/s, ~1.5× swing). A branch mispredict is
  ~15–20 cycles, but each iteration already costs hundreds of cycles of
  interpreter dispatch, so the branch barely registers.
- **PyPy** swings the most — ~4.5× from ~181 (p=0.5) to ~770–820 (extremes).
  Its compiled loop is so tight (a guarded `float_ne` + `float_add`) that
  misprediction cost dominates the per-element budget.
- **V8** is in between (~2.3× swing). Its indexed loop is fast but a bit heavier
  per element than PyPy's, so the branch is a smaller fraction of the total.
- At **p = 0.5** PyPy and V8 nearly converge (181 vs 166) — PyPy's lead exists
  only when the branch is predictable. The curves are roughly symmetric; both
  JITs are a hair faster at high p(NaN) (the common path is a bare `continue`,
  slightly cheaper than the `+=`).

A side note on JIT warm-up: running the whole sweep *in one process* instead
makes PyPy's curve **asymmetric** (high-p drops to ~400 instead of ~800),
because the loop is compiled once on the first fraction (p=0.05, "not-NaN
common") and that trace is reused for every later fraction, so high p(NaN)
constantly fails the guard into the side-trace. Measuring each fraction in a
fresh process (as above) isolates the p(NaN) effect from that warm-once
history. Pass fractions as CLI args to `sweep_nansum.{py,js}` to do so.

### Methodology note: idiomatic loops

Each loop uses its language's idiomatic form, not a literal translation: an
indexed `for (let i ...)` in JS, `for v in values` in Python. This matters a
lot for V8 — array `for...of` carries iterator-protocol overhead V8 doesn't
elide, so the same plain-sum loop written with `for...of` runs at only
~213 M/s (≈8× slower) instead of ~1706 M/s. PyPy's tracing JIT optimizes the
iterator away either way, so its numbers are unaffected by the loop shape.
We use the indexed form in JS because that's the fair, locally-idiomatic
comparison.

To inspect PyPy's compiled traces, dump them with `PYPYLOG`:

```sh
PYPYLOG=jit-log-opt:traces/poly.log pypy3 benchmark_poly.py
```
