"use strict";
// JS port of benchmark_mono.py: polysum over 5M summands, all one type
// (monomorphic call site), for comparison with benchmark_poly.js.
const { AddInt, polysum } = require("./summands");
const { makeRng, timeBest, report } = require("./bench_util");

const N = 5_000_000;
const REPEATS = 5;
const SEED = 1234;

function buildData(n) {
  const rng = makeRng(SEED);
  const data = new Array(n);
  for (let i = 0; i < n; i++) data[i] = new AddInt(Math.floor(rng() * 10));
  return data;
}

const data = buildData(N);
polysum(data); // warm-up
const { best, result } = timeBest(() => polysum(data), REPEATS);
report({
  label: "polysum, monomorphic (all AddInt)",
  n: N,
  unit: "calls/s",
  best,
  result,
});
