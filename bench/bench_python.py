#!/usr/bin/env python3
"""Python bench harness. Prints one JSON line per run.

    PYTHONPATH=python python3 bench/bench_python.py rules
    PYTHONPATH=python python3 bench/bench_python.py features
    PYTHONPATH=python python3 bench/bench_python.py batch
    PYTHONPATH=python python3 bench/bench_python.py neural MODEL TOK LABELS

Modes rules/features time each of the 15 fixture notes individually
(2 warm-up passes discarded, 7 measured passes; pooled per-note
latencies -> median + p95 by nearest-rank). Batch times redact_batch
over the 1,000-doc fixture-derived corpus (1 warm-up, 5 reps).
Neural loads the real v9 int8 model and times full redacts per note.
"""
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))

import phi_redact  # noqa: E402

NOTES_PATH = os.path.join(HERE, "..", "e2e", "parity", "notes.jsonl")
CORPUS_PATH = os.path.join(HERE, "corpus.jsonl")

FEATURE_KWARGS = dict(
    operator="hash",
    salt="bench",
    denylist=["Maria"],
    recognizers=[
        phi_redact.PatternRecognizer(
            label="EMPLOYEE_ID", patterns=[r"EMP\d{6}"], score=0.9),
        phi_redact.PatternRecognizer(
            label="REF_CODE", patterns=[r"[A-Z]{2}-\d{4}"], score=0.8),
    ],
    thresholds={"REF_CODE": 0.9},
    review_margin=0.2,
)


def load_notes():
    with open(NOTES_PATH) as f:
        return [json.loads(line)["note"] for line in f if line.strip()]


def load_corpus():
    with open(CORPUS_PATH) as f:
        return [json.loads(line)["note"] for line in f if line.strip()]


def pct95(samples):
    s = sorted(samples)
    return s[max(0, int(0.95 * len(s) + 0.999999) - 1)]


def per_note(fn, notes, warmups=2, reps=7):
    for _ in range(warmups):
        for n in notes:
            fn(n)
    samples = []
    for _ in range(reps):
        for n in notes:
            t0 = time.perf_counter_ns()
            fn(n)
            samples.append((time.perf_counter_ns() - t0) / 1e6)
    return {
        "median_ms": round(statistics.median(samples), 4),
        "p95_ms": round(pct95(samples), 4),
        "samples": len(samples),
        "notes": len(notes),
    }


def main() -> int:
    mode = sys.argv[1]
    notes = load_notes()
    if mode == "rules":
        r = phi_redact.Redactor()
        out = per_note(lambda n: r.redact(n), notes)
    elif mode == "features":
        r = phi_redact.Redactor(**FEATURE_KWARGS)
        out = per_note(lambda n: r.redact(n), notes)
    elif mode == "batch":
        corpus = load_corpus()
        r = phi_redact.Redactor()
        for _ in range(1):
            r.redact_batch(corpus)
        totals = []
        for _ in range(5):
            t0 = time.perf_counter_ns()
            r.redact_batch(corpus)
            totals.append((time.perf_counter_ns() - t0) / 1e9)
        med = statistics.median(totals)
        out = {
            "docs": len(corpus),
            "median_total_s": round(med, 4),
            "notes_per_sec": round(len(corpus) / med, 1),
            "per_note_ms": round(med / len(corpus) * 1000, 4),
            "reps": len(totals),
        }
    elif mode == "neural":
        model, tok, labels = sys.argv[2], sys.argv[3], sys.argv[4]
        t0 = time.perf_counter_ns()
        r = phi_redact.Redactor(model=model, tokenizer=tok, labels=labels)
        load_s = (time.perf_counter_ns() - t0) / 1e9
        out = per_note(lambda n: r.redact(n), notes)
        out["model_load_s"] = round(load_s, 3)
    else:
        print(f"unknown mode {mode}", file=sys.stderr)
        return 2
    out.update({"language": "python", "mode": mode})
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
