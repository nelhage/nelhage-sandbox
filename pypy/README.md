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
- `bench_util.js` — shared JS helpers (seeded PRNG + best-of-N timer).
- `run_benchmark.sh` — runs `benchmark.py` under CPython and PyPy.

Each benchmark times 5 reps and reports the best, after one warm-up call so the
JIT is compiled before timing. The loops use `for v in values` / `for (const v
of values)` so the Python and JS versions iterate the same way (see the V8
caveat below).

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
| plain `sum` (10M floats)       | ~85 M/s      | ~1870 M/s | ~213 M/s          |
| `nansum`, 10% random NaN       | ~52 M/s      | ~536 M/s  | ~134 M/s          |
| `polysum`, monomorphic         | ~33 M/s      | ~334 M/s  | ~213 M/s          |
| `polysum`, polymorphic (3-way) | ~24 M/s      | ~117 M/s  | ~86 M/s           |

(Throughput is M elements/s for the sums, M calls/s for `polysum`. Engines
each use their own seeded data, so the checksums differ across languages but
are stable within an engine.)

## What the numbers show

**Both JITs (PyPy, V8) crush the CPython interpreter, and the more a loop is
*dispatch* rather than *arithmetic*, the more the JIT lead narrows.**

- **Plain float-add** is PyPy's best case — its tracing JIT compiles the hot
  loop to a bare `float_add` over an unboxed-double list (~22× over CPython).
- **nansum** adds a `v != v` test; for PyPy it compiles to a cheap `float_ne` +
  guard, but 10% *random* NaNs trip that guard into a side-trace (bridge)
  unpredictably, cutting its lead to ~10×.
- **Monomorphic dispatch** still lets the JIT inline the single `add` target
  behind a class guard that never fails.
- **Polymorphic dispatch** is the hardest case for both JITs: with a random
  3-way type mix the receiver-type guard / inline cache misses constantly, so
  most iterations fall back to a real method lookup+call. PyPy drops ~2.9×
  going mono→poly (334→117 M/s) and V8 drops ~2.5× (213→86 M/s), while CPython —
  already paying full per-call dispatch regardless of type — loses only ~1.4×
  (33→24 M/s). At the polymorphic call site PyPy and V8 end up about even.

### Caveat: V8 `for...of` vs indexed loops

The faithful port uses `for (const v of values)` to match Python's
`for v in values`. **V8 does not optimize array `for...of` nearly as well as an
indexed loop** — for the plain sum, switching to `for (let i = 0; i < a.length;
i++)` takes V8 from ~213 M/s to **~1700 M/s**, essentially matching PyPy. PyPy's
tracing JIT, by contrast, optimizes the iterator away regardless. So the large
PyPy-over-V8 gap on the arithmetic loops is mostly an iteration-protocol
artifact, not a fundamental codegen difference; the dispatch benchmarks (where
per-iteration work dominates the iterator overhead) are the more apples-to-apples
comparison.

To inspect PyPy's compiled traces, dump them with `PYPYLOG`:

```sh
PYPYLOG=jit-log-opt:traces/poly.log pypy3 benchmark_poly.py
```
