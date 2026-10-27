// Cross-SDK parity runner for the 0.4.0 + 0.5.0 features (TypeScript side).
// Usage: node run_ts_features.mjs notes.jsonl
// Prints JSON: [{ name, results: [{ redacted, entities, mapping, counts, review }] }]
// The config list below must stay in lockstep with test_parity.py's
// FEATURE_CONFIGS — same names, same semantics. "batch_hash" and
// "batch_recognizer_review" exercise redactBatch(); every other config
// calls redact() per note.
// Requires: cd ../typescript && npm install && npm run build

import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const { redact, redactBatch } = require(
  join(here, "..", "..", "typescript", "dist", "index.js"),
);

const notesPath = process.argv[2];
if (!notesPath) {
  console.error("usage: node run_ts_features.mjs notes.jsonl");
  process.exit(2);
}

const notes = [];
for (const line of readFileSync(notesPath, "utf8").split("\n")) {
  if (!line.trim()) continue;
  notes.push(JSON.parse(line).note);
}

const CONFIGS = [
  ["label", { operator: "label" }],
  ["mask", { operator: "mask" }],
  ["hash_salted", { operator: "hash", salt: "parity-salt" }],
  ["hash_unsalted", { operator: "hash" }],
  ["remove", { operator: "remove" }],
  ["allowlist", { allowlist: ["415-555-0132", "jane@example.com"] }],
  ["denylist", { denylist: ["Maria", "Jane"] }],
  ["denylist_label", { denylist: ["Santos"], denyLabel: "NAME_PATIENT" }],
  [
    "allow_and_deny",
    { allowlist: ["415-555-0132"], denylist: ["Maria"] },
  ],
  // 0.5.0: custom recognizers, per-label thresholds, review mode.
  [
    "recognizer_basic",
    {
      recognizers: [
        { label: "EMPLOYEE_ID", patterns: ["EMP\\d{6}"], score: 0.9 },
      ],
    },
  ],
  [
    "recognizer_context",
    {
      recognizers: [
        {
          label: "REF_CODE",
          patterns: ["[A-Z]{2}-\\d{4}"],
          score: 0.5,
          context: ["ref"],
          contextBoost: 0.45,
        },
      ],
    },
  ],
  [
    "recognizer_require_context",
    {
      recognizers: [
        {
          label: "REF_CODE",
          patterns: ["[A-Z]{2}-\\d{4}"],
          score: 0.9,
          context: ["ref"],
          requireContext: true,
        },
      ],
    },
  ],
  [
    "recognizer_validator",
    {
      recognizers: [
        {
          label: "MEMBER_ID",
          patterns: ["\\b\\d{9}\\b"],
          score: 0.9,
          validator: "aba",
        },
      ],
    },
  ],
  [
    "thresholds_lower",
    {
      recognizers: [
        { label: "REF_CODE", patterns: ["[A-Z]{2}-\\d{4}"], score: 0.5 },
      ],
      thresholds: { REF_CODE: 0.4 },
    },
  ],
  [
    "thresholds_raise_review",
    {
      recognizers: [
        { label: "REF_CODE", patterns: ["[A-Z]{2}-\\d{4}"], score: 0.8 },
      ],
      thresholds: { REF_CODE: 0.9 },
      reviewMargin: 0.2,
    },
  ],
  [
    "review_margin_global",
    {
      recognizers: [
        { label: "REF_CODE", patterns: ["[A-Z]{2}-\\d{4}"], score: 0.55 },
      ],
      reviewMargin: 0.1,
    },
  ],
];

const shape = (r) => ({
  redacted: r.redacted,
  entities: r.entities.map((e) => ({
    start: e.start,
    end: e.end,
    label: e.label,
    source: e.source,
    placeholder: e.placeholder ?? null,
    score: e.score ?? null,
  })),
  mapping: r.mapping,
  counts: r.counts,
  review: r.review.map((c) => ({
    start: c.start,
    end: c.end,
    label: c.label,
    text: c.text,
    score: c.score,
    source: c.source,
  })),
});

const out = CONFIGS.map(([name, opts]) => ({
  name,
  results: notes.map((note) => shape(redact(note, [], opts))),
}));
out.push({
  name: "batch_hash",
  results: redactBatch(notes, { operator: "hash", salt: "parity-salt" })
    .map(shape),
});
out.push({
  name: "batch_recognizer_review",
  results: redactBatch(notes, {
    recognizers: [
      { label: "REF_CODE", patterns: ["[A-Z]{2}-\\d{4}"], score: 0.8 },
    ],
    thresholds: { REF_CODE: 0.9 },
    reviewMargin: 0.2,
  }).map(shape),
});

console.log(JSON.stringify(out));
