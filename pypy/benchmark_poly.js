"use strict";
// JS port of benchmark_poly.py: polysum over 5M summands, a random mix of the
// three types -- a polymorphic call site V8 can't collapse to one target.
const { KINDS, polysum } = require("./summands");
const { makeRng, timeBest, report } = require("./bench_util");

const N = 5_000_000;
const REPEATS = 5;
const SEED = 1234;

function buildData(n) {
  const rng = makeRng(SEED);
  const data = new Array(n);
  for (let i = 0; i < n; i++) {
    const Kind = KINDS[Math.floor(rng() * KINDS.length)];
    data[i] = new Kind(Math.floor(rng() * 10));
  }
  return data;
}

const data = buildData(N);
polysum(data); // warm-up
const { best, result } = timeBest(() => polysum(data), REPEATS);
report({
  label: `polysum, polymorphic (${KINDS.length}-way)`,
  n: N,
  unit: "calls/s",
  best,
  result,
});
