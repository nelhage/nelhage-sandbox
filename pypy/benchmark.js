"use strict";
// JS port of benchmark.py: plain float sum over a 10M-element array.
const { sumFloats } = require("./sum_floats");
const { timeBest, report } = require("./bench_util");

const N = 10_000_000;
const REPEATS = 5;

function buildData(n) {
  const data = new Array(n);
  for (let i = 0; i < n; i++) data[i] = i * 0.5;
  return data;
}

const data = buildData(N);
sumFloats(data); // warm-up: let V8 (TurboFan) optimize the loop before timing
const { best, result } = timeBest(() => sumFloats(data), REPEATS);
report({ label: "plain sum (10M floats)", n: N, unit: "elements/s", best, result });
