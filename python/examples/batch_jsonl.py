#!/usr/bin/env python3
"""JSONL batch redaction — the enterprise batch-pipeline entry point.

Reads JSONL records (one JSON object per line) from a file or stdin,
redacts the text field of each record with ONE loaded Redactor, and
writes JSONL to a file or stdout: the record with its text field
replaced by the redacted text, plus a "counts" object (label -> n).

    python3 examples/batch_jsonl.py notes.jsonl redacted.jsonl
    cat notes.jsonl | python3 examples/batch_jsonl.py > redacted.jsonl
    python3 examples/batch_jsonl.py --operator hash --salt "$SALT" in.jsonl

With --model/--tokenizer the neural layer joins in (needs the onnx
extra); without them it is rules-only and dependency-free. Records
whose text field is missing or not a string pass through untouched.
Blank lines are skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Allow running from a repo checkout without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))

import phi_redact  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", nargs="?", default="-",
                    help="input JSONL file (default: stdin)")
    ap.add_argument("output", nargs="?", default="-",
                    help="output JSONL file (default: stdout)")
    ap.add_argument("--field", default="text",
                    help="record field holding the text (default: text)")
    ap.add_argument("--model", help="ONNX model path (enables neural)")
    ap.add_argument("--tokenizer", help="tokenizer.json path")
    ap.add_argument("--operator", default="numbered",
                    choices=phi_redact.OPERATORS,
                    help="anonymize operator (default: numbered)")
    ap.add_argument("--salt", default="", help="salt for --operator hash")
    args = ap.parse_args(argv)

    redactor = phi_redact.Redactor(
        model=args.model, tokenizer=args.tokenizer,
        operator=args.operator, salt=args.salt)

    fin = sys.stdin if args.input == "-" else open(args.input)
    fout = sys.stdout if args.output == "-" else open(args.output, "w")
    try:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = record.get(args.field)
            if isinstance(text, str):
                result = redactor.redact(text)
                record[args.field] = result.redacted
                record["counts"] = result.counts
            fout.write(json.dumps(record) + "\n")
    finally:
        if fin is not sys.stdin:
            fin.close()
        if fout is not sys.stdout:
            fout.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
