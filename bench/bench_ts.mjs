// TypeScript/Node bench harness. Prints one JSON line per run.
//
//   node bench/bench_ts.mjs rules
//   node bench/bench_ts.mjs features
//   node bench/bench_ts.mjs batch
//
// Same method as bench_python.py: per-note timing over the 15 parity
// fixture notes (2 warm-up passes, 7 measured; pooled latencies,
// median + nearest-rank p95); batch over the 1,000-doc fixture-derived
// corpus (1 warm-up, 5 reps). The local TS SDK has no neural engine —
// there is deliberately no "neural" mode here (see PERF.md).

import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const { redact, redactBatch } = require(
  join(here, "..", "typescript", "dist", "index.js"),
);

const notes = readFileSync(join(here, "..", "e2e", "parity", "notes.jsonl"), "utf8")
  .split("\n")
  .filter((l) => l.trim())
  .map((l) => JSON.parse(l).note);
const corpus = readFileSync(join(here, "corpus.jsonl"), "utf8")
  .split("\n")
  .filter((l) => l.trim())
  .map((l) => JSON.parse(l).note);

const FEATURE_OPTS = {
  operator: "hash",
  salt: "bench",
  denylist: ["Maria"],
  recognizers: [
    { label: "EMPLOYEE_ID", patterns: ["EMP\\d{6}"], score: 0.9 },
    { label: "REF_CODE", patterns: ["[A-Z]{2}-\\d{4}"], score: 0.8 },
  ],
  thresholds: { REF_CODE: 0.9 },
  reviewMargin: 0.2,
};

const median = (xs) => {
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
};
const p95 = (xs) => {
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.max(0, Math.ceil(0.95 * s.length) - 1)];
};
const round4 = (x) => Math.round(x * 10000) / 10000;

function perNote(fn, warmups = 2, reps = 7) {
  for (let w = 0; w < warmups; w++) for (const n of notes) fn(n);
  const samples = [];
  for (let r = 0; r < reps; r++) {
    for (const n of notes) {
      const t0 = process.hrtime.bigint();
      fn(n);
      samples.push(Number(process.hrtime.bigint() - t0) / 1e6);
    }
  }
  return {
    median_ms: round4(median(samples)),
    p95_ms: round4(p95(samples)),
    samples: samples.length,
    notes: notes.length,
  };
}

const mode = process.argv[2];
let out;
if (mode === "rules") {
  out = perNote((n) => redact(n));
} else if (mode === "features") {
  out = perNote((n) => redact(n, [], FEATURE_OPTS));
} else if (mode === "batch") {
  redactBatch(corpus);
  const totals = [];
  for (let r = 0; r < 5; r++) {
    const t0 = process.hrtime.bigint();
    redactBatch(corpus);
    totals.push(Number(process.hrtime.bigint() - t0) / 1e9);
  }
  const med = median(totals);
  out = {
    docs: corpus.length,
    median_total_s: round4(med),
    notes_per_sec: Math.round((corpus.length / med) * 10) / 10,
    per_note_ms: round4((med / corpus.length) * 1000),
    reps: totals.length,
  };
} else {
  console.error(`unknown mode ${mode}`);
  process.exit(2);
}
console.log(JSON.stringify({ language: "typescript", mode, ...out }));
