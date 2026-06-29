"use strict";
// Sweep nansum throughput across a range of p(NaN). Mirrors sweep_nansum.py.
//     node sweep_nansum.js
const { nansumFloats } = require("./sum_floats");
const { makeRng } = require("./bench_util");

const N = 10_000_000;
const REPEATS = 5;
const SEED = 1234;
const DEFAULT_FRACTIONS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95];
// Pass fractions as CLI args to sweep just those (e.g. one per process).
const cliFractions = process.argv.slice(2).map(Number);
const FRACTIONS = cliFractions.length ? cliFractions : DEFAULT_FRACTIONS;

function buildData(n, frac) {
  const rng = makeRng(SEED);
  const data = new Array(n);
  for (let i = 0; i < n; i++) data[i] = i * 0.5;
  for (let i = 0; i < n; i++) if (rng() < frac) data[i] = NaN;
  return data;
}

function bestTime(data) {
  nansumFloats(data); // warm-up for this dataset
  let best = Infinity;
  for (let r = 0; r < REPEATS; r++) {
    const start = process.hrtime.bigint();
    nansumFloats(data);
    const elapsed = Number(process.hrtime.bigint() - start) / 1e9;
    if (elapsed < best) best = elapsed;
  }
  return best;
}

console.log(
  `V8 ${process.versions.v8} / Node ${process.versions.node}  ` +
    `(N=${N.toLocaleString()}, ${REPEATS} reps, best)`
);
console.log(`  ${"p(NaN)".padStart(7)} ${"best (s)".padStart(10)} ${"M elem/s".padStart(10)}`);
for (const frac of FRACTIONS) {
  const data = buildData(N, frac);
  const best = bestTime(data);
  console.log(
    `  ${frac.toFixed(2).padStart(7)} ${best.toFixed(4).padStart(10)} ` +
      `${(N / best / 1e6).toFixed(1).padStart(10)}`
  );
}
