#!/usr/bin/env node
/**
 * JSONL batch redaction — the enterprise batch-pipeline entry point.
 *
 * Reads JSONL records (one JSON object per line) from a file or stdin,
 * redacts the text field of each record, and writes JSONL to a file or
 * stdout: the record with its text field replaced by the redacted
 * text, plus a "counts" object (label -> n).
 *
 *   node examples/batch_jsonl.mjs notes.jsonl redacted.jsonl
 *   cat notes.jsonl | node examples/batch_jsonl.mjs > redacted.jsonl
 *   node examples/batch_jsonl.mjs --operator hash --salt "$SALT" in.jsonl
 *
 * Run `npm run build` first (this imports ../dist/index.js). The local
 * engine is rules-only; the neural layer runs through the phi-redact
 * server (see RemoteRedactor). Records whose text field is missing or
 * not a string pass through untouched. Blank lines are skipped.
 */

import { readFileSync, writeFileSync } from "node:fs";
import { redact } from "../dist/index.js";

function parseArgs(argv) {
  const args = { field: "text", operator: "numbered", salt: "" };
  const positional = [];
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--field") args.field = argv[++i];
    else if (a === "--operator") args.operator = argv[++i];
    else if (a === "--salt") args.salt = argv[++i];
    else positional.push(a);
  }
  args.input = positional[0] ?? "-";
  args.output = positional[1] ?? "-";
  return args;
}

const args = parseArgs(process.argv.slice(2));
const input = args.input === "-"
  ? readFileSync(0, "utf8")
  : readFileSync(args.input, "utf8");

const outLines = [];
for (const line of input.split("\n")) {
  const trimmed = line.trim();
  if (!trimmed) continue;
  const record = JSON.parse(trimmed);
  const text = record[args.field];
  if (typeof text === "string") {
    const result = redact(text, [], {
      operator: args.operator,
      salt: args.salt,
    });
    record[args.field] = result.redacted;
    record.counts = result.counts;
  }
  outLines.push(JSON.stringify(record));
}

const outText = outLines.length ? outLines.join("\n") + "\n" : "";
if (args.output === "-") process.stdout.write(outText);
else writeFileSync(args.output, outText);
