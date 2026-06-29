"use strict";
// JS port of benchmark.py: plain integer sum over a 10M-element array.
const { sumInts } = require("./sums");
const { timeBest, report } = require("./bench_util");

const N = 10_000_000;
const REPEATS = 5;

function buildData(n) {
  const data = new Array(n);
  for (let i = 0; i < n; i++) data[i] = i;
  return data;
}

const data = buildData(N);
sumInts(data); // warm-up: let V8 (TurboFan) optimize the loop before timing
const { best, result } = timeBest(() => sumInts(data), REPEATS);
report({ label: "plain sum (10M ints)", n: N, unit: "elements/s", best, result });
