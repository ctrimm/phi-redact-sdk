// Cross-SDK parity runner (TypeScript side).
// Usage: node run_ts.mjs notes.jsonl
// Prints a JSON array: [{ redacted, entities: [{start,end,label}], mapping }]
// Requires: cd ../typescript && npm install && npm run build

import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const { redact } = require(join(here, "..", "..", "typescript", "dist", "index.js"));

const notesPath = process.argv[2];
if (!notesPath) {
  console.error("usage: node run_ts.mjs notes.jsonl");
  process.exit(2);
}

const out = [];
for (const line of readFileSync(notesPath, "utf8").split("\n")) {
  if (!line.trim()) continue;
  const { note } = JSON.parse(line);
  const r = redact(note);
  out.push({
    redacted: r.redacted,
    entities: r.entities.map((e) => ({
      start: e.start,
      end: e.end,
      label: e.label,
    })),
    mapping: r.mapping,
  });
}
console.log(JSON.stringify(out));
