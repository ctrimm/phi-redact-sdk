// Cross-SDK parity runner for the 0.4.0 features (TypeScript side).
// Usage: node run_ts_features.mjs notes.jsonl
// Prints JSON: [{ name, results: [{ redacted, entities, mapping, counts }] }]
// The config list below must stay in lockstep with test_parity.py's
// FEATURE_CONFIGS — same names, same semantics. "batch_hash" exercises
// redactBatch(); every other config calls redact() per note.
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
];

const shape = (r) => ({
  redacted: r.redacted,
  entities: r.entities.map((e) => ({
    start: e.start,
    end: e.end,
    label: e.label,
    source: e.source,
    placeholder: e.placeholder ?? null,
  })),
  mapping: r.mapping,
  counts: r.counts,
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

console.log(JSON.stringify(out));
