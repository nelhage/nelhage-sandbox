"use strict";
// JS port of sum_floats.py: a plain summing loop and an nansum (skip-NaN) loop.
// Uses `for ... of` to mirror Python's `for v in values`.

function sumFloats(values) {
  let total = 0.0;
  for (const v of values) {
    total += v;
  }
  return total;
}

function nansumFloats(values) {
  // nansum semantics: skip NaN. `v !== v` is true only for NaN.
  let total = 0.0;
  for (const v of values) {
    if (v !== v) continue;
    total += v;
  }
  return total;
}

module.exports = { sumFloats, nansumFloats };
