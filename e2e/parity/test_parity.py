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


# ---------------------------------------------------------------------------
# Feature parity (0.4.0 operators/lists/batch; 0.5.0 recognizers,
# thresholds, review). Config names/semantics must stay in lockstep
# with run_ts_features.mjs.
# ---------------------------------------------------------------------------

FEATURE_CONFIGS = {
    "label": {"operator": "label"},
    "mask": {"operator": "mask"},
    "hash_salted": {"operator": "hash", "salt": "parity-salt"},
    "hash_unsalted": {"operator": "hash"},
    "remove": {"operator": "remove"},
    "allowlist": {"allowlist": ["415-555-0132", "jane@example.com"]},
    "denylist": {"denylist": ["Maria", "Jane"]},
    "denylist_label": {"denylist": ["Santos"], "deny_label": "NAME_PATIENT"},
    "allow_and_deny": {"allowlist": ["415-555-0132"],
                       "denylist": ["Maria"]},
    # 0.5.0: custom recognizers, per-label thresholds, review mode.
    "recognizer_basic": {"recognizers": [
        phi_redact.PatternRecognizer(
            label="EMPLOYEE_ID", patterns=[r"EMP\d{6}"], score=0.9)]},
    "recognizer_context": {"recognizers": [
        phi_redact.PatternRecognizer(
            label="REF_CODE", patterns=[r"[A-Z]{2}-\d{4}"], score=0.5,
            context=["ref"], context_boost=0.45)]},
    "recognizer_require_context": {"recognizers": [
        phi_redact.PatternRecognizer(
            label="REF_CODE", patterns=[r"[A-Z]{2}-\d{4}"], score=0.9,
            context=["ref"], require_context=True)]},
    "recognizer_validator": {"recognizers": [
        phi_redact.PatternRecognizer(
            label="MEMBER_ID", patterns=[r"\b\d{9}\b"], score=0.9,
            validator="aba")]},
    "thresholds_lower": {
        "recognizers": [
            phi_redact.PatternRecognizer(
                label="REF_CODE", patterns=[r"[A-Z]{2}-\d{4}"],
                score=0.5)],
        "thresholds": {"REF_CODE": 0.4}},
    "thresholds_raise_review": {
        "recognizers": [
            phi_redact.PatternRecognizer(
                label="REF_CODE", patterns=[r"[A-Z]{2}-\d{4}"],
                score=0.8)],
        "thresholds": {"REF_CODE": 0.9}, "review_margin": 0.2},
    "review_margin_global": {
        "recognizers": [
            phi_redact.PatternRecognizer(
                label="REF_CODE", patterns=[r"[A-Z]{2}-\d{4}"],
                score=0.55)],
        "review_margin": 0.1},
}


def _ts_feature_results():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not on PATH")
    dist = os.path.join(ROOT, "typescript", "dist", "index.js")
    if not os.path.exists(dist):
        pytest.skip("typescript/dist not built; run: cd typescript && npm run build")
    src = os.path.join(ROOT, "typescript", "src", "engine.ts")
    compiled = os.path.join(ROOT, "typescript", "dist", "engine.js")
    if os.path.getmtime(src) > os.path.getmtime(compiled):
        pytest.fail("typescript/dist is stale (src/engine.ts changed); "
                    "run: cd typescript && npm run build")
    proc = subprocess.run(
        [node, os.path.join(HERE, "run_ts_features.mjs"),
         os.path.join(HERE, "notes.jsonl")],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, \
        f"run_ts_features.mjs failed: {proc.stderr[:500]}"
    return json.loads(proc.stdout)


def _shape(result):
    return {
        "redacted": result.redacted,
        "entities": [
            {"start": e["start"], "end": e["end"], "label": e["label"],
             "source": e["source"], "placeholder": e.get("placeholder"),
             "score": e.get("score")}
            for e in result.entities
        ],
        "mapping": result.mapping,
        "counts": result.counts,
        "review": [
            {"start": c["start"], "end": c["end"], "label": c["label"],
             "text": c["text"], "score": c["score"],
             "source": c["source"]}
            for c in result.review
        ],
    }


def test_feature_parity():
    notes = _notes()
    ts_out = _ts_feature_results()
    assert [c["name"] for c in ts_out] == \
        list(FEATURE_CONFIGS) + ["batch_hash", "batch_recognizer_review"]
    for config in ts_out:
        assert len(config["results"]) == len(notes)
        if config["name"] == "batch_hash":
            batch = phi_redact.Redactor(
                operator="hash", salt="parity-salt").redact_batch(notes)
            py_results = [_shape(r) for r in batch]
        elif config["name"] == "batch_recognizer_review":
            batch = phi_redact.Redactor(
                recognizers=[phi_redact.PatternRecognizer(
                    label="REF_CODE", patterns=[r"[A-Z]{2}-\d{4}"],
                    score=0.8)],
                thresholds={"REF_CODE": 0.9},
                review_margin=0.2).redact_batch(notes)
            py_results = [_shape(r) for r in batch]
        else:
            opts = FEATURE_CONFIGS[config["name"]]
            py_results = [_shape(phi_redact.redact(n, **opts))
                          for n in notes]
        for note, ts, py in zip(notes, config["results"], py_results):
            assert ts == py, \
                f"{config['name']} differs for note {note!r}:\nTS: {ts}\nPY: {py}"
