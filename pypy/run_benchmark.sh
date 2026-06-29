#!/usr/bin/env bash
# Run the sum_floats benchmark under both CPython and PyPy and show both results.
# Assumes the direnv/flake environment is active (python3 and pypy3 on PATH).
set -euo pipefail
cd "$(dirname "$0")"

echo "=== CPython ==="
python3 benchmark.py
echo
echo "=== PyPy ==="
pypy3 benchmark.py
