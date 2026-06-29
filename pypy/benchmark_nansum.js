"use strict";
// JS port of benchmark_nansum.py: nansum over 10M floats, ~10% randomly NaN.
const { nansumFloats } = require("./sum_floats");
const { makeRng, timeBest, report } = require("./bench_util");

const N = 10_000_000;
const REPEATS = 5;
const NAN_FRACTION = 0.1;
const SEED = 1234;

function buildData(n) {
  const rng = makeRng(SEED);
  const data = new Array(n);
  for (let i = 0; i < n; i++) data[i] = i * 0.5;
  for (let i = 0; i < n; i++) {
    if (rng() < NAN_FRACTION) data[i] = NaN;
  }
  return data;
}

const data = buildData(N);
let nNan = 0;
for (const v of data) if (v !== v) nNan++;

nansumFloats(data); // warm-up
const { best, result } = timeBest(() => nansumFloats(data), REPEATS);
report({
  label: `nansum, ${((100 * nNan) / N).toFixed(1)}% random NaN`,
  n: N,
  unit: "elements/s",
  best,
  result,
});
