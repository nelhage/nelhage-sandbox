"use strict";
// JS port of sums.py: a plain integer sum and a sentinel-skipping sum.
// Indexed loops (the idiomatic, well-optimized shape in V8).

const SENTINEL = -1;

function sumInts(values) {
  let total = 0;
  for (let i = 0; i < values.length; i++) {
    total += values[i];
  }
  return total;
}

function sentinelSum(values) {
  // Skip the -1 sentinel; sum everything else. Data values are all >= 0.
  let total = 0;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (v === SENTINEL) continue;
    total += v;
  }
  return total;
}

module.exports = { SENTINEL, sumInts, sentinelSum };
