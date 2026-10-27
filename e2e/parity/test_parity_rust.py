"""Cross-SDK parity: Rust vs Python (the reference implementation).

Mirrors test_parity.py's TS checks for the Rust SDK
(../../rust, crate phi-redact). The Rust side runs through
rust/examples/parity_runner.rs, which prints exactly the shapes
run_ts.mjs / run_ts_features.mjs print. Skips when cargo is not
available; builds the runner in release mode on first use (cargo
itself handles staleness after that).

Run:  python -m pytest e2e/parity/test_parity_rust.py -q
"""
import json
import os
import shutil
import subprocess

import pytest

import phi_redact

from test_parity import FEATURE_CONFIGS, _notes, _shape

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RUST_DIR = os.path.join(ROOT, "rust")
RUNNER = os.path.join(
    RUST_DIR, "target", "release", "examples", "parity_runner")


def _cargo_env():
    env = dict(os.environ)
    tooling = os.path.expanduser("~/workspace/tooling/rust")
    if os.path.isdir(os.path.join(tooling, "cargo", "bin")):
        env.setdefault("RUSTUP_HOME", os.path.join(tooling, "rustup"))
        env.setdefault("CARGO_HOME", os.path.join(tooling, "cargo"))
        env["PATH"] = os.path.join(tooling, "cargo", "bin") + \
            os.pathsep + env.get("PATH", "")
    return env


def _rust_out(mode):
    cargo = shutil.which("cargo", path=_cargo_env()["PATH"])
    if cargo is None:
        pytest.skip("cargo not on PATH")
    build = subprocess.run(
        [cargo, "build", "--release", "--example", "parity_runner"],
        cwd=RUST_DIR, env=_cargo_env(), capture_output=True, text=True,
        timeout=900)
    assert build.returncode == 0, \
        f"cargo build failed: {build.stderr[-500:]}"
    proc = subprocess.run(
        [RUNNER, mode, os.path.join(HERE, "notes.jsonl")],
        capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, \
        f"parity_runner {mode} failed: {proc.stderr[:500]}"
    return json.loads(proc.stdout)


def test_rust_basic_parity():
    notes = _notes()
    rs_out = _rust_out("basic")
    assert len(rs_out) == len(notes)
    for note, rs in zip(notes, rs_out):
        py = phi_redact.redact(note)
        py_ents = [
            {"start": e["start"], "end": e["end"], "label": e["label"]}
            for e in py.entities
        ]
        assert rs["redacted"] == py.redacted, \
            f"redacted text differs: {note!r}"
        assert rs["entities"] == py_ents, \
            f"entity spans differ: {note!r}"
        assert rs["mapping"] == py.mapping, f"mapping differs: {note!r}"
        assert phi_redact.restore(rs["redacted"], rs["mapping"]) == note


def test_rust_feature_parity():
    notes = _notes()
    rs_out = _rust_out("features")
    assert [c["name"] for c in rs_out] == \
        list(FEATURE_CONFIGS) + ["batch_hash", "batch_recognizer_review"]
    for config in rs_out:
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
        for note, rs, py in zip(notes, config["results"], py_results):
            assert rs == py, (
                f"{config['name']} differs for note {note!r}:\n"
                f"RUST: {rs}\nPY: {py}")
