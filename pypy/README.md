# CPython vs PyPy vs V8 vs C: pure-language loop benchmarks

A small harness comparing equivalent summing loops on four engines —
CPython, PyPy, V8 (Node), and optimized C — across a few workloads that
stress their compilers/JITs differently.

## Environment

All engines come from the `pypy` devShell in the repo's `flake.nix`
(`pkgs.python3` for CPython, `pkgs.pypy3` for PyPy, `pkgs.nodejs` for V8,
`pkgs.gcc`/`pkgs.gnumake` for C). The `.envrc` activates it via direnv, so
`python3`, `pypy3`, `node`, and `gcc` are all on `PATH` inside this directory.

## Files

The Python and JavaScript ports mirror each other one-to-one; the C versions
(in `c/`) mirror them in turn.

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
- `make_report.py` — generates `report.html`, a self-contained HTML summary
  with an inline-SVG line plot of the p(NaN) sweep.
- `c/` — the C port: `sum_floats.{c,h}`, `summands.{c,h}` (polymorphic dispatch
  via a function-pointer vtable — each concrete type embeds a `summand_base`
  with a `summand_fn *func` as its first member and casts in its function),
  `bench_util.h` (PRNG + timer), `benchmark.c` (all four headline benchmarks),
  `sweep_nansum.c`, and a `Makefile`. Build with `make` in `c/`.

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
( cd c && make && ./benchmark )    # all four, in C
```

## Results (rough, one machine, x86_64)

| Benchmark                      | CPython 3.13 | PyPy 3.11 | V8 13.6 / Node 24 | C (gcc -O2) |
|--------------------------------|--------------|-----------|-------------------|-------------|
| plain `sum` (10M floats)       | ~85 M/s      | ~1870 M/s | ~1706 M/s         | ~2237 M/s   |
| `nansum`, 10% random NaN       | ~52 M/s      | ~536 M/s  | ~275 M/s          | ~748 M/s    |
| `polysum`, monomorphic         | ~33 M/s      | ~334 M/s  | ~618 M/s          | ~680 M/s    |
| `polysum`, polymorphic (3-way) | ~24 M/s      | ~117 M/s  | ~156 M/s          | ~182 M/s    |

(Throughput is M elements/s for the sums, M calls/s for `polysum`. The C and JS
builds share the same seeded-data RNG, so their checksums match exactly;
CPython/PyPy use Python's RNG so theirs differ but are stable per engine.)

## What the numbers show

**Optimized C is fastest on every row — but the margins are small (~1.2–1.4×
over the better JIT on most rows), and on polymorphic dispatch C's lead is only
~1.2×.** The JITs are remarkably close to native scalar C; the interpreter
(CPython) is the real outlier, 20–40× behind.

- **Plain float-add** — C ~2237, PyPy ~1870, V8 ~1706, all within ~1.3×. At
  `-O2` C keeps strict IEEE semantics, so this stays a *scalar* reduction (no
  vectorization — FP add isn't associative); the JITs likewise emit a bare
  scalar add over an unboxed-double array, and all three are near
  memory-bandwidth bound. Building C with `-ffast-math -march=native` lets it
  vectorize the reduction and jumps to **~3340 M/s**.
- **nansum** — C ~748, PyPy ~536, V8 ~275. The `v != v` test + 10% *random*
  NaNs is an unpredictable branch that hurts everyone; among the JITs PyPy's
  trace+bridge handling absorbs it better than V8.
- **Monomorphic dispatch** — C ~680 and V8 ~618 are close; V8's hidden-class +
  inline-cache machinery inlines the single `add` and nearly matches a native
  indirect call. PyPy ~334 trails here.
- **Polymorphic dispatch** — C ~182, V8 ~156, PyPy ~117. With a random 3-way
  mix *everyone* pays an indirect-branch / inline-cache miss per call, which is
  why even C falls to ~182: the bottleneck is the CPU's indirect-branch
  predictor, not the language. C drops ~3.7× mono→poly (680→182), V8 ~4.0×
  (618→156), PyPy ~2.9× (334→117), CPython only ~1.4× (it already paid full
  dispatch cost).

The takeaway: among the JITs there's no single winner — PyPy's tracing JIT is
strongest on tight numeric loops, V8's inline-cache design wins on method
dispatch — and both sit within a small factor of optimized C until you reach
the polymorphic case, where the hardware branch predictor caps everyone.

## nansum across p(NaN): a branch-prediction curve

Sweeping the NaN fraction (each point measured in its own process, so the JIT
compiles fresh for that fraction) gives throughput in M elem/s:

| p(NaN) | CPython 3.13 | PyPy 3.11 | V8 13.6 / Node 24 | C (gcc -O2) |
|--------|--------------|-----------|-------------------|-------------|
| 0.05   | ~56          | ~768      | ~311              | ~1199       |
| 0.10   | ~58          | ~545      | ~275              | ~754        |
| 0.20   | ~56          | ~345      | ~230              | ~477        |
| 0.30   | ~53          | ~255      | ~202              | ~372        |
| 0.40   | ~44          | ~203      | ~178              | ~322        |
| 0.50   | ~51          | ~181      | ~166              | ~295        |
| 0.60   | ~54          | ~191      | ~176              | ~318        |
| 0.70   | ~57          | ~255      | ~201              | ~366        |
| 0.80   | ~56          | ~345      | ~240              | ~457        |
| 0.90   | ~63          | ~556      | ~316              | ~705        |
| 0.95   | ~66          | ~821      | ~385              | ~1122       |

All four trace a **U**: throughput bottoms out near **p = 0.5** and recovers
toward both extremes. That's the signature of **branch (mis)prediction** on the
`if v != v` test — at p ≈ 0.5 the branch is maximally unpredictable (~50%
mispredict rate), while at p → 0 or p → 1 it's almost always the same way and
the CPU predicts it for free.

- **CPython** is nearly flat (~44–66 M/s, ~1.5× swing). A branch mispredict is
  ~15–20 cycles, but each iteration already costs hundreds of cycles of
  interpreter dispatch, so the branch barely registers.
- **C** and **PyPy** swing the most (~4×): C from ~295 (p=0.5) to ~1100–1200
  (extremes), PyPy from ~181 to ~770–820. Their loops are so tight (a compare +
  conditional + add) that the mispredict dominates the per-element budget.
- **V8** is in between (~2.3× swing) — its loop is a bit heavier per element, so
  the branch is a smaller fraction of the total.
- At **p = 0.5** the mispredict bottlenecks everyone: even C drops to ~295, and
  PyPy and V8 nearly converge (181 vs 166). The curves are roughly symmetric
  (the C and JIT loops are a hair faster at high p(NaN), where the common path
  is a bare skip rather than a `+=`). Note these are fresh-process measurements;
  see the warm-up note below for why PyPy's *in-process* curve looks asymmetric.

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
