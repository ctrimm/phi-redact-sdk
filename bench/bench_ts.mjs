// TypeScript/Node bench harness. Prints one JSON line per invocation.
//
//   node bench/bench_ts.mjs scan|rules|features
//   node bench/bench_ts.mjs batch|batch-features
//
// Driven by bench/run_bench.py, which interleaves languages
// round-by-round; this harness executes BENCH_REPS measured passes
// (after BENCH_WARMUPS discarded passes) in-process and reports raw
// per-note samples plus SHA-256 digests of its inputs and outputs in
// the payload format shared with the Python and Rust harnesses (see
// bench/bench_python.py). The local TS SDK has no neural engine —
// there is deliberately no "neural" mode here.

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const { redact, redactBatch, scan } = require(
  join(here, "..", "typescript", "dist", "index.js"),
);

const notesPath = process.env.BENCH_NOTES_PATH ??
  join(here, "..", "e2e", "parity", "notes.jsonl");
const notes = readFileSync(notesPath, "utf8")
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

const WARMUPS = Number(process.env.BENCH_WARMUPS ?? 2);
const REPS = Number(process.env.BENCH_REPS ?? 7);
const BATCH_WARMUPS = Number(process.env.BENCH_WARMUPS ?? 1);
const BATCH_REPS = Number(process.env.BENCH_REPS ?? 5);

const scoreStr = (score, withScore) =>
  !withScore || score === undefined || score === null
    ? ""
    : score.toFixed(4);

function resultPayload(r, withScore) {
  const lines = [r.redacted];
  for (const e of r.entities) {
    lines.push(
      `${e.start},${e.end},${e.label},${scoreStr(e.score, withScore)},` +
        `${e.source},${e.placeholder ?? ""}`,
    );
  }
  for (const c of r.review) {
    lines.push(
      `R${c.start},${c.end},${c.label},${scoreStr(c.score, withScore)},` +
        `${c.source}`,
    );
  }
  return lines.join("\n");
}

function spansPayload(spans, withScore) {
  return spans
    .map(
      (e) =>
        `${e.start},${e.end},${e.label},${scoreStr(e.score, withScore)},` +
        `${e.source},${e.placeholder ?? ""}`,
    )
    .join("\n");
}

function digestPayloads(payloads) {
  const h = createHash("sha256");
  for (const p of payloads) {
    h.update(p, "utf8");
    h.update(Buffer.from([0]));
  }
  return h.digest("hex");
}

function inputDigest(texts) {
  const h = createHash("sha256");
  for (const t of texts) {
    h.update(t, "utf8");
    h.update("\n");
  }
  return h.digest("hex");
}

const round4 = (x) => Math.round(x * 10000) / 10000;

function perNote(fn) {
  for (let w = 0; w < WARMUPS; w++) for (const n of notes) fn(n);
  const samples = [];
  for (let r = 0; r < REPS; r++) {
    for (const n of notes) {
      const t0 = process.hrtime.bigint();
      fn(n);
      samples.push(round4(Number(process.hrtime.bigint() - t0) / 1e6));
    }
  }
  return samples;
}

const mode = process.argv[2];
let out;
if (mode === "scan") {
  // Digest pass first (untimed full pass -> steady state).
  const digest = digestPayloads(
    notes.map((n) => spansPayload(scan(n), true)),
  );
  const digestNs = digestPayloads(
    notes.map((n) => spansPayload(scan(n), false)),
  );
  out = { samples_ms: perNote((n) => scan(n)), digest, digest_noscore: digestNs };
} else if (mode === "rules" || mode === "features") {
  const opts = mode === "rules" ? {} : FEATURE_OPTS;
  const fn = (n) => redact(n, [], opts);
  const digest = digestPayloads(notes.map((n) => resultPayload(fn(n), true)));
  const digestNs = digestPayloads(
    notes.map((n) => resultPayload(fn(n), false)),
  );
  out = { samples_ms: perNote(fn), digest, digest_noscore: digestNs };
} else if (mode === "batch" || mode === "batch-features") {
  const opts = mode === "batch" ? {} : FEATURE_OPTS;
  // Digest from a dedicated untimed pass (not tied to warmups, so
  // BENCH_WARMUPS=0 invocations still report it).
  const digestResults = redactBatch(corpus, opts);
  const digest = digestPayloads(
    digestResults.map((r) => resultPayload(r, true)),
  );
  const digestNs = digestPayloads(
    digestResults.map((r) => resultPayload(r, false)),
  );
  for (let w = 0; w < BATCH_WARMUPS; w++) redactBatch(corpus, opts);
  const totals = [];
  for (let r = 0; r < BATCH_REPS; r++) {
    const t0 = process.hrtime.bigint();
    redactBatch(corpus, opts);
    totals.push(round4(Number(process.hrtime.bigint() - t0) / 1e9));
  }
  console.log(
    JSON.stringify({
      language: "typescript",
      mode,
      docs: corpus.length,
      totals_s: totals,
      digest,
      digest_noscore: digestNs,
      input_digest: inputDigest(corpus),
    }),
  );
  process.exit(0);
} else {
  console.error(`unknown mode ${mode}`);
  process.exit(2);
}
console.log(
  JSON.stringify({
    language: "typescript",
    mode,
    input_digest: inputDigest(notes),
    notes: notes.length,
    ...out,
  }),
);
