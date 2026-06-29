# PyPy vs CPython: float-summing benchmark

A tiny harness comparing a pure-Python summing loop on CPython and PyPy.

## Environment

Both interpreters come from the `pypy` devShell in the repo's `flake.nix`
(`pkgs.python3` for CPython, `pkgs.pypy3` for PyPy). The `.envrc` activates it
via direnv, so `python3` and `pypy3` are both on `PATH` inside this directory.

## Files

- `sum_floats.py` — the function under test: sums a list of floats with an
  explicit Python `for` loop.
- `benchmark.py` — times the loop over a 10M-element list (5 reps, best time
  reported) and prints interpreter, throughput, and a checksum.
- `run_benchmark.sh` — runs `benchmark.py` under both interpreters.

## Running

```sh
./run_benchmark.sh
# or individually:
python3 benchmark.py
pypy3 benchmark.py
```

## Result (rough, one machine, x86_64)

| Interpreter   | Best time (10M sum) | Throughput      |
|---------------|---------------------|-----------------|
| CPython 3.13  | ~0.118 s            | ~85 M elem/s    |
| PyPy 3.11     | ~0.005 s            | ~1870 M elem/s  |

PyPy's tracing JIT compiles the hot loop to native code, making it roughly
**20×** faster here. CPython interprets each bytecode op per iteration.
The harness warms up once before timing so PyPy's JIT is compiled by then.
