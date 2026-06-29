# PyPy vs CPython: pure-Python loop benchmarks

A small harness comparing pure-Python loops on CPython and PyPy across a few
workloads that stress the JIT differently.

## Environment

Both interpreters come from the `pypy` devShell in the repo's `flake.nix`
(`pkgs.python3` for CPython, `pkgs.pypy3` for PyPy). The `.envrc` activates it
via direnv, so `python3` and `pypy3` are both on `PATH` inside this directory.

## Files

- `sum_floats.py` — the functions under test: `sum_floats` (plain summing loop)
  and `nansum_floats` (skip-NaN summing loop).
- `summands.py` — a `Summand` base class with three subclasses (`AddInt`,
  `SquareInt`, `Identity`) and `polysum`, which loops calling `s.add(total)`.
- `benchmark.py` — plain float sum over a 10M-element list.
- `benchmark_nansum.py` — nansum over 10M floats with ~10% randomly NaN.
- `benchmark_poly.py` — `polysum` over 5M summands, a **random mix** of the
  three types (polymorphic call site).
- `benchmark_mono.py` — `polysum` over 5M summands, all one type
  (monomorphic call site), for comparison with the polymorphic case.
- `run_benchmark.sh` — runs `benchmark.py` under both interpreters.

Each benchmark times 5 reps and reports the best, after one warm-up call so
PyPy's JIT is compiled before timing.

## Running

```sh
./run_benchmark.sh                 # plain sum, both interpreters
# or any benchmark under either interpreter:
python3 benchmark_poly.py
pypy3   benchmark_poly.py
```

## Results (rough, one machine, x86_64)

| Benchmark                     | CPython 3.13 | PyPy 3.11   | PyPy speedup |
|-------------------------------|--------------|-------------|--------------|
| plain `sum` (10M floats)      | ~85 M/s      | ~1870 M/s   | ~22×         |
| `nansum`, 10% random NaN      | ~52 M/s      | ~536 M/s    | ~10×         |
| `polysum`, monomorphic        | ~33 M/s      | ~334 M/s    | ~10×         |
| `polysum`, polymorphic (3-way)| ~24 M/s      | ~117 M/s    | ~5×          |

(Throughput is M elements/s for the sums, M calls/s for `polysum`.)

## What the numbers show

The more a loop is *dispatch* rather than *arithmetic*, the smaller PyPy's
lead:

- **Plain float-add** is PyPy's best case — the JIT compiles the hot loop to a
  bare `float_add` on unboxed doubles (~22×).
- **nansum** adds a `v != v` test that compiles to a cheap `float_ne` + guard,
  but 10% *random* NaNs trip that guard into a side-trace (bridge) ~1-in-10
  unpredictably, dragging the speedup to ~10×.
- **Monomorphic dispatch** still lets the JIT inline the single `add` target
  behind a class guard that never fails (~10×).
- **Polymorphic dispatch** is the hardest: with a random 3-way type mix the
  receiver-class guard fails ~2/3 of the time, so most iterations deopt into
  bridges doing a real method lookup+call. PyPy's lead compresses to ~5×.
  CPython was already paying full per-call dispatch cost regardless of type,
  so it's relatively less penalized — note PyPy loses ~2.9× going mono→poly
  (334→117 M/s) while CPython loses only ~1.4× (33→24 M/s).

To inspect the compiled traces, dump them with `PYPYLOG`:

```sh
PYPYLOG=jit-log-opt:traces/poly.log pypy3 benchmark_poly.py
```
