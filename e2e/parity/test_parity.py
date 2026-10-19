"""E2E cross-SDK parity: Python and TypeScript SDKs must produce identical
redaction output for the same inputs.

Runs the TypeScript SDK via node (needs typescript/dist built) and the
Python SDK in-process, then compares redacted text, entity spans, and
mappings note by note. Any drift in the rules ports fails loudly here.
"""

import json
import os
import shutil
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "python"))

import phi_redact  # noqa: E402


def _notes():
    notes = []
    with open(os.path.join(HERE, "notes.jsonl")) as f:
        for line in f:
            line = line.strip()
            if line:
                notes.append(json.loads(line)["note"])
    return notes


def _ts_results(notes):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not on PATH")
    dist = os.path.join(ROOT, "typescript", "dist", "index.js")
    if not os.path.exists(dist):
        pytest.skip("typescript/dist not built; run: cd typescript && npm run build")
    src = os.path.join(ROOT, "typescript", "src", "rules.ts")
    compiled = os.path.join(ROOT, "typescript", "dist", "rules.js")
    if os.path.getmtime(src) > os.path.getmtime(compiled):
        pytest.fail("typescript/dist is stale (src/rules.ts changed); "
                    "run: cd typescript && npm run build")
    proc = subprocess.run(
        [node, os.path.join(HERE, "run_ts.mjs"),
         os.path.join(HERE, "notes.jsonl")],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, f"run_ts.mjs failed: {proc.stderr[:500]}"
    return json.loads(proc.stdout)


def test_parity():
    notes = _notes()
    ts_out = _ts_results(notes)
    assert len(ts_out) == len(notes)
    for note, ts in zip(notes, ts_out):
        py = phi_redact.redact(note)
        py_ents = [
            {"start": e["start"], "end": e["end"], "label": e["label"]}
            for e in py.entities
        ]
        assert ts["redacted"] == py.redacted, f"redacted text differs: {note!r}"
        assert ts["entities"] == py_ents, f"entity spans differ: {note!r}"
        assert ts["mapping"] == py.mapping, f"mapping differs: {note!r}"
        # both sides restore to the original
        assert phi_redact.restore(py.redacted, py.mapping) == note
