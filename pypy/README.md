# CPython vs PyPy vs V8 vs C: pure-language loop benchmarks

A small harness comparing equivalent **integer** summing loops on four engines
— CPython, PyPy, V8 (Node), and optimized C — across a few workloads that
stress their compilers/JITs differently. Everything operates on `int`, so the
plain sum, the sentinel-skipping sum, and the vtable-dispatch sums are all
directly comparable.

## Environment

All engines come from the `pypy` devShell in the repo's `flake.nix`
(`pkgs.python3` for CPython, `pkgs.pypy3` for PyPy, `pkgs.nodejs` for V8,
`pkgs.gcc`/`pkgs.gnumake` for C). The `.envrc` activates it via direnv, so
`python3`, `pypy3`, `node`, and `gcc` are all on `PATH` inside this directory.

## Files

The Python and JavaScript ports mirror each other one-to-one; the C versions
(in `c/`) mirror them in turn.

- `sums.{py,js}` — the functions under test: a plain int summing loop and a
  sentinel-skipping sum (skip a `-1` sentinel; the int analogue of the old
  skip-NaN loop).
- `summands.{py,js}` — a `Summand` base class with three subclasses (`AddInt`,
  `SquareInt`, `Identity`) and `polysum`, which loops calling `s.add(total)`.
- `benchmark.{py,js}` — plain int sum over a 10M-element list.
- `benchmark_sentinel.{py,js}` — sentinel sum over 10M ints with ~10% randomly
  set to the `-1` sentinel.
- `benchmark_poly.{py,js}` — `polysum` over 5M summands, a **random mix** of
  the three types (polymorphic call site).
- `benchmark_mono.{py,js}` — `polysum` over 5M summands, all one type
  (monomorphic call site), for comparison with the polymorphic case.
- `sweep_sentinel.{py,js}` — sweep sentinel-sum throughput across a range of
  p(sentinel). With no args it sweeps a default set of fractions in one process;
  pass fractions as CLI args (e.g. `sweep_sentinel.py 0.5`) to measure one per
  process and avoid the JIT reusing a single early-compiled trace.
- `bench_util.js` — shared JS helpers (seeded PRNG + best-of-N timer).
- `run_benchmark.sh` — runs `benchmark.py` under CPython and PyPy.
- `make_report.py` — generates `report.html`, a self-contained HTML summary
  with an inline-SVG line plot of the p(sentinel) sweep.
- `c/` — the C port: `sums.{c,h}`, `summands.{c,h}` (polymorphic dispatch via a
  function-pointer vtable — each concrete type embeds a `summand_base` with a
  `summand_fn *func` as its first member and casts in its function),
  `bench_util.h` (PRNG + timer), `benchmark.c` (all four headline benchmarks),
  `sweep_sentinel.c`, and a `Makefile`. Build with `make` in `c/`.

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
| plain `sum` (10M ints)         | ~55 M/s      | ~1665 M/s | ~1274 M/s         | ~3846 M/s   |
| `sentinel sum`, 10% skipped    | ~44 M/s      | ~660 M/s  | ~357 M/s          | ~2353 M/s   |
| `polysum`, monomorphic         | ~33 M/s      | ~330 M/s  | ~597 M/s          | ~698 M/s    |
| `polysum`, polymorphic (3-way) | ~24 M/s      | ~117 M/s  | ~156 M/s          | ~181 M/s    |

(Throughput is M elements/s for the sums, M calls/s for `polysum`. The C and JS
builds share the same seeded-data RNG, so their checksums match exactly;
CPython/PyPy use Python's RNG so theirs differ but are stable per engine.)

## What the numbers show

**Optimized C is fastest on every row. On the two scalar sums its lead is now
large (~2–3×) because the int loops fit C's strengths; on dispatch the JITs
close back to within ~1.2×.** CPython remains the outlier, 20–70× behind.

- **Plain int-add** — C ~3846, PyPy ~1665, V8 ~1274. At `-O2` gcc emits **no
  SIMD** (auto-vectorization is `-O3`); C wins anyway because an integer add is
  1-cycle latency, so the serial accumulator chain runs ~1 add/cycle — versus
  the ~3–4-cycle latency that capped the old *float* sum. (At `-O3` gcc
  vectorizes to **~5490 M/s**.) The JITs do machine-int adds too but carry more
  per-iteration overhead; V8 also promotes the running total from a 31-bit SMI
  to a boxed double once it exceeds ~2³¹, which costs it here.
- **Sentinel sum** — C ~2353, PyPy ~660, V8 ~357. gcc **if-converts** the
  `if v == -1: continue` into a branchless `cmov`, so C pays no misprediction at
  all (see the sweep below). The JITs emit a real branch and eat the ~10%
  mispredicts.
- **Monomorphic dispatch** — C ~698 and V8 ~597 are close; V8's hidden-class +
  inline-cache machinery inlines the single `add` and nearly matches a native
  indirect call. PyPy ~330 trails here.
- **Polymorphic dispatch** — C ~181, V8 ~156, PyPy ~117. With a random 3-way mix
  *everyone* pays an indirect-branch / inline-cache miss per call, so even C
  falls to ~181: the bottleneck is the CPU's indirect-branch predictor, not the
  language. C drops ~3.9× mono→poly (698→181), V8 ~3.8× (597→156), PyPy ~2.8×
  (330→117), CPython only ~1.4× (it already paid full dispatch cost).

The takeaway: C's lead is large exactly where the compiler can apply a trick the
JITs don't (1-cycle int reduction, branchless `cmov`), and shrinks to almost
nothing on polymorphic dispatch, where the hardware branch predictor caps
everyone. Among the JITs, PyPy still wins the tight numeric loops and V8 the
method dispatch.

## sentinel sum across p(sentinel): four different curves

Sweeping the sentinel fraction (each point measured in its own process, so the
JIT compiles fresh for that fraction) gives throughput in M elem/s:

| p(sentinel) | CPython 3.13 | PyPy 3.11 | V8 13.6 / Node 24 | C (gcc -O2) |
|-------------|--------------|-----------|-------------------|-------------|
| 0.05        | ~44          | ~1007     | ~408              | ~2276       |
| 0.10        | ~44          | ~665      | ~361              | ~2309       |
| 0.20        | ~45          | ~421      | ~295              | ~2316       |
| 0.30        | ~45          | ~314      | ~255              | ~2309       |
| 0.40        | ~46          | ~249      | ~228              | ~2310       |
| 0.50        | ~47          | ~217      | ~218              | ~2304       |
| 0.60        | ~52          | ~228      | ~230              | ~2293       |
| 0.70        | ~59          | ~284      | ~258              | ~2314       |
| 0.80        | ~68          | ~400      | ~340              | ~2160       |
| 0.90        | ~81          | ~665      | ~517              | ~2305       |
| 0.95        | ~89          | ~993      | ~737              | ~2303       |

Two effects compete here: a **branch-misprediction** penalty that peaks at
p = 0.5 (the `if v == -1` skip is maximally unpredictable there), and a
**less-work** effect — more sentinels mean fewer adds. Each engine weights them
differently, so unlike the old float sweep there's no single shared shape:

- **C is dead flat** (~2300 across the whole range). gcc if-converts the skip
  into a branchless `cmov` that runs every iteration regardless of the data, so
  there's no branch to mispredict *and* no work saved by skipping — the
  optimizer erased the entire effect. (The lone ~2160 dip at p=0.8 is noise.)
- **PyPy traces a symmetric U** (~1007 → ~217 → ~993, ~4.6× swing). It emits a
  real guard, so misprediction dominates and bottoms out at p = 0.5; the add is
  cheap enough (~1 cycle) that the less-work effect barely tilts it.
- **V8 is a tilted U** — a dip at ~0.5 from misprediction, but the high-skip end
  (~737) runs well above the low-skip end (~408) because it does fewer (boxed)
  adds. Both effects are visible.
- **CPython rises monotonically** (~44 → ~89). A mispredict is ~15–20 cycles,
  but each `total += v` costs far more (it allocates a new int object), so the
  branch is invisible and the only thing that shows is doing fewer adds as the
  skip rate climbs.

A side note on JIT warm-up: running the whole sweep *in one process* makes
PyPy's curve **asymmetric** (the high-skip end collapses), because the loop is
compiled once on the first fraction (p=0.05, "rarely skip") and that trace is
reused for every later fraction, so a high sentinel rate constantly fails the
guard into the side-trace. Measuring each fraction in a fresh process (as above)
isolates the p(sentinel) effect from that warm-once history. Pass fractions as
CLI args to `sweep_sentinel.{py,js}` to do so.

### Methodology note: idiomatic loops

Each loop uses its language's idiomatic form, not a literal translation: an
indexed `for (let i ...)` in JS, `for v in values` in Python. This matters a
lot for V8 — array `for...of` carries iterator-protocol overhead V8 doesn't
elide, so the same plain int-sum loop written with `for...of` runs at only
~217 M/s (≈6× slower) instead of ~1295 M/s. PyPy's tracing JIT optimizes the
iterator away either way, so its numbers are unaffected by the loop shape.
We use the indexed form in JS because that's the fair, locally-idiomatic
comparison.

To inspect PyPy's compiled traces, dump them with `PYPYLOG`:

```sh
PYPYLOG=jit-log-opt:traces/poly.log pypy3 benchmark_poly.py
```
