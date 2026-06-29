"use strict";
// JS port of benchmark_sentinel.py: sum 10M ints, skipping a -1 sentinel
// placed at ~10% of positions at random.
const { SENTINEL, sentinelSum } = require("./sums");
const { makeRng, timeBest, report } = require("./bench_util");

const N = 10_000_000;
const REPEATS = 5;
const SENTINEL_FRACTION = 0.1;
const SEED = 1234;

function buildData(n) {
  const rng = makeRng(SEED);
  const data = new Array(n);
  for (let i = 0; i < n; i++) data[i] = i;
  for (let i = 0; i < n; i++) {
    if (rng() < SENTINEL_FRACTION) data[i] = SENTINEL;
  }
  return data;
}

const data = buildData(N);
let nSkip = 0;
for (const v of data) if (v === SENTINEL) nSkip++;

sentinelSum(data); // warm-up
const { best, result } = timeBest(() => sentinelSum(data), REPEATS);
report({
  label: `sentinel sum, ${((100 * nSkip) / N).toFixed(1)}% skipped`,
  n: N,
  unit: "elements/s",
  best,
  result,
});
