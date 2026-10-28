#!/usr/bin/env python3
"""Python bench harness. Prints one JSON line per invocation.

    PYTHONPATH=python python3 bench/bench_python.py scan|rules|features
    PYTHONPATH=python python3 bench/bench_python.py batch|batch-features
    PYTHONPATH=python python3 bench/bench_python.py neural MODEL TOK LABELS
    PYTHONPATH=python python3 bench/bench_python.py neural-stages MODEL TOK LABELS

Driven by bench/run_bench.py, which interleaves languages
round-by-round; this harness executes BENCH_REPS measured passes
(after BENCH_WARMUPS discarded passes) in-process and reports raw
per-note samples plus SHA-256 digests of its inputs and outputs, so
the driver can verify all three SDKs read byte-identical corpora
and produce identical outputs under identical configs. The digest
payload format is shared with the TS and Rust harnesses:

  per redact result: redacted text, then one line per entity
  "start,end,label,score,source,placeholder", then one line per
  review candidate "Rstart,end,label,score,source"; score is ""
  when the span carries none, else formatted %.4f. Payloads are
  joined per note and hashed incrementally with a NUL separator.
  digest_noscore is the same payload with every score blanked.
"""
import hashlib
import json
import os
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

WARMUPS = int(os.environ.get("BENCH_WARMUPS", "2"))
REPS = int(os.environ.get("BENCH_REPS", "7"))
BATCH_WARMUPS = int(os.environ.get("BENCH_WARMUPS", "1"))
BATCH_REPS = int(os.environ.get("BENCH_REPS", "5"))


def load_notes():
    with open(NOTES_PATH) as f:
        return [json.loads(line)["note"] for line in f if line.strip()]


def load_corpus():
    with open(CORPUS_PATH) as f:
        return [json.loads(line)["note"] for line in f if line.strip()]


def _score(score, with_score):
    if not with_score or score is None:
        return ""
    return f"{score:.4f}"


def result_payload(result, with_score):
    d = result.to_dict()
    lines = [d["redacted"]]
    for e in d["entities"]:
        lines.append(
            f"{e['start']},{e['end']},{e['label']},"
            f"{_score(e.get('score'), with_score)},{e['source']},"
            f"{e.get('placeholder') or ''}")
    for c in d["review"]:
        lines.append(
            f"R{c['start']},{c['end']},{c['label']},"
            f"{_score(c.get('score'), with_score)},{c['source']}")
    return "\n".join(lines)


def spans_payload(spans, with_score):
    return "\n".join(
        f"{e['start']},{e['end']},{e['label']},"
        f"{_score(e.get('score'), with_score)},{e['source']},"
        f"{e.get('placeholder') or ''}"
        for e in spans)


def digest_payloads(payloads):
    h = hashlib.sha256()
    for p in payloads:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def input_digest(texts):
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def per_note(fn, notes):
    """Warm up, then time fn per note; returns raw ms samples."""
    for _ in range(WARMUPS):
        for n in notes:
            fn(n)
    samples = []
    for _ in range(REPS):
        for n in notes:
            t0 = time.perf_counter_ns()
            fn(n)
            samples.append((time.perf_counter_ns() - t0) / 1e6)
    return [round(s, 4) for s in samples]


def redact_mode(mode, notes):
    # Digest pass first (untimed): it is also a full warm pass, so
    # every invocation reaches steady state before timing even when
    # BENCH_WARMUPS=0.
    if mode == "scan":
        fn = phi_redact.scan
        digest = digest_payloads(
            [spans_payload(fn(n), True) for n in notes])
        digest_ns = digest_payloads(
            [spans_payload(fn(n), False) for n in notes])
        return {"samples_ms": per_note(fn, notes),
                "digest": digest, "digest_noscore": digest_ns}
    r = (phi_redact.Redactor() if mode == "rules"
         else phi_redact.Redactor(**FEATURE_KWARGS))
    digest = digest_payloads(
        [result_payload(r.redact(n), True) for n in notes])
    digest_ns = digest_payloads(
        [result_payload(r.redact(n), False) for n in notes])
    return {"samples_ms": per_note(r.redact, notes),
            "digest": digest, "digest_noscore": digest_ns}


def batch_mode(mode):
    corpus = load_corpus()
    r = (phi_redact.Redactor() if mode == "batch"
         else phi_redact.Redactor(**FEATURE_KWARGS))
    # Digest from a dedicated untimed pass (not tied to warmups, so
    # BENCH_WARMUPS=0 invocations still report it).
    results = r.redact_batch(corpus)
    digest = digest_payloads([result_payload(x, True) for x in results])
    digest_ns = digest_payloads(
        [result_payload(x, False) for x in results])
    for _ in range(BATCH_WARMUPS):
        r.redact_batch(corpus)
    totals = []
    for _ in range(BATCH_REPS):
        t0 = time.perf_counter_ns()
        r.redact_batch(corpus)
        totals.append((time.perf_counter_ns() - t0) / 1e9)
    return {
        "docs": len(corpus),
        "totals_s": [round(t, 4) for t in totals],
        "digest": digest,
        "digest_noscore": digest_ns,
        "input_digest": input_digest(corpus),
    }


class _TimedTokenizer:
    """Delegating wrapper that accumulates encode() time. Swapped
    onto the real OnnxEngine so stage timings come from the actual
    predict() code path, not a re-implementation."""

    def __init__(self, tok):
        object.__setattr__(self, "_tok", tok)
        object.__setattr__(self, "time_ns", 0)

    def encode(self, *args, **kwargs):
        t0 = time.perf_counter_ns()
        out = self._tok.encode(*args, **kwargs)
        object.__setattr__(
            self, "time_ns",
            self.time_ns + (time.perf_counter_ns() - t0))
        return out

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_tok"), name)


class _TimedSession:
    """Same trick for the ONNX session: accumulates run() time."""

    def __init__(self, sess):
        object.__setattr__(self, "_sess", sess)
        object.__setattr__(self, "time_ns", 0)

    def run(self, *args, **kwargs):
        t0 = time.perf_counter_ns()
        out = self._sess.run(*args, **kwargs)
        object.__setattr__(
            self, "time_ns",
            self.time_ns + (time.perf_counter_ns() - t0))
        return out

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_sess"), name)


def neural_stages(notes, model, tok_path, labels):
    r = phi_redact.Redactor(model=model, tokenizer=tok_path, labels=labels)
    eng = r.neural
    timed_tok = _TimedTokenizer(eng.tok)
    timed_sess = _TimedSession(eng.sess)
    eng.tok, eng.sess = timed_tok, timed_sess

    def stages_for(note):
        tok0, sess0 = timed_tok.time_ns, timed_sess.time_ns
        t0 = time.perf_counter_ns()
        spans = eng.predict(note)
        total = (time.perf_counter_ns() - t0) / 1e6
        tok_ms = (timed_tok.time_ns - tok0) / 1e6
        inf_ms = (timed_sess.time_ns - sess0) / 1e6
        # post = everything else inside predict(): feed assembly,
        # softmax/argmax, BIOES decode, dedupe.
        return spans, tok_ms, inf_ms, max(total - tok_ms - inf_ms, 0.0)

    # Digest pass first (untimed full pass -> steady state).
    digest = digest_payloads(
        [spans_payload(eng.predict(n), True) for n in notes])
    digest_ns = digest_payloads(
        [spans_payload(eng.predict(n), False) for n in notes])
    for _ in range(WARMUPS):
        for n in notes:
            stages_for(n)
    tok_s, inf_s, post_s, tot_s = [], [], [], []
    for _ in range(REPS):
        for n in notes:
            _spans, a, b, c = stages_for(n)
            tok_s.append(round(a, 4))
            inf_s.append(round(b, 4))
            post_s.append(round(c, 4))
            tot_s.append(round(a + b + c, 4))
    return {
        "stages_ms": {"tokenize": tok_s, "inference": inf_s, "post": post_s},
        "samples_ms": tot_s,
        "digest": digest,
        "digest_noscore": digest_ns,
    }


def main() -> int:
    mode = sys.argv[1]
    notes = load_notes()
    if mode in ("scan", "rules", "features"):
        out = redact_mode(mode, notes)
    elif mode in ("batch", "batch-features"):
        print(json.dumps({"language": "python", "mode": mode,
                          **batch_mode(mode)}))
        return 0
    elif mode == "neural":
        model, tok, labels = sys.argv[2], sys.argv[3], sys.argv[4]
        t0 = time.perf_counter_ns()
        r = phi_redact.Redactor(model=model, tokenizer=tok, labels=labels)
        load_s = (time.perf_counter_ns() - t0) / 1e9
        digest = digest_payloads(
            [result_payload(r.redact(n), True) for n in notes])
        digest_ns = digest_payloads(
            [result_payload(r.redact(n), False) for n in notes])
        out = {
            "samples_ms": per_note(r.redact, notes),
            "digest": digest,
            "digest_noscore": digest_ns,
            "model_load_s": round(load_s, 3),
        }
    elif mode == "neural-stages":
        model, tok, labels = sys.argv[2], sys.argv[3], sys.argv[4]
        t0 = time.perf_counter_ns()
        phi_redact.Redactor(model=model, tokenizer=tok, labels=labels)
        load_s = (time.perf_counter_ns() - t0) / 1e9
        out = {**neural_stages(notes, model, tok, labels),
               "model_load_s": round(load_s, 3)}
    else:
        print(f"unknown mode {mode}", file=sys.stderr)
        return 2
    out.update({"language": "python", "mode": mode,
                "input_digest": input_digest(notes), "notes": len(notes)})
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
