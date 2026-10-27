#!/usr/bin/env python3
"""Build the bench corpus: the 15 parity fixture notes, repeated and
lightly perturbed to ~1,000 documents for batch-throughput runs.

Fixture-derived by construction (this is a throughput corpus, not an
evaluation set): each document is one fixture note plus a varying
"Log entry N." suffix, in a seeded shuffle so consecutive documents
differ. The suffix carries no PII-shaped tokens.

    python3 bench/gen_corpus.py [count] > bench/corpus.jsonl
"""
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NOTES = os.path.join(HERE, "..", "e2e", "parity", "notes.jsonl")


def main() -> int:
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    notes = []
    with open(NOTES) as f:
        for line in f:
            line = line.strip()
            if line:
                notes.append(json.loads(line)["note"])
    rng = random.Random(20261027)
    order = list(range(count))
    rng.shuffle(order)
    out = []
    for k in order:
        base = notes[k % len(notes)]
        out.append({"note": f"{base} Log entry {k}."})
    for doc in out:
        print(json.dumps(doc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
