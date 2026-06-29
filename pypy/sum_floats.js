"use strict";
// JS port of sum_floats.py: a plain summing loop and an nansum (skip-NaN) loop.
// Uses an indexed `for` loop -- the idiomatic, well-optimized shape in V8
// (array `for...of` carries iterator-protocol overhead V8 doesn't elide).

function sumFloats(values) {
  let total = 0.0;
  for (let i = 0; i < values.length; i++) {
    total += values[i];
  }
  return total;
}

function nansumFloats(values) {
  // nansum semantics: skip NaN. `v !== v` is true only for NaN.
  let total = 0.0;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (v !== v) continue;
    total += v;
  }
  return total;
}

module.exports = { sumFloats, nansumFloats };
