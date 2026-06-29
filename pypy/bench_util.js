"use strict";
// Shared helpers for the JS benchmarks: a seeded PRNG and a best-of-N timer.

// mulberry32: a small deterministic PRNG so data is reproducible across runs.
// (JS has no seedable Math.random.) Returns floats in [0, 1).
function makeRng(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Run fn() `repeats` times, returning the best (fastest) wall time in seconds.
function timeBest(fn, repeats) {
  let best = Infinity;
  let result;
  for (let i = 0; i < repeats; i++) {
    const start = process.hrtime.bigint();
    result = fn();
    const elapsed = Number(process.hrtime.bigint() - start) / 1e9;
    if (elapsed < best) best = elapsed;
  }
  return { best, result };
}

function report({ label, n, unit, best, result }) {
  const rate = n / best;
  console.log(`V8 ${process.versions.v8} / Node ${process.versions.node}`);
  console.log(`  ${label}`);
  console.log(`  count       : ${n.toLocaleString()}`);
  console.log(`  best time   : ${best.toFixed(4)} s`);
  console.log(`  throughput  : ${(rate / 1e6).toFixed(1)} M ${unit}`);
  console.log(`  (sum check) : ${result}`);
}

module.exports = { makeRng, timeBest, report };
