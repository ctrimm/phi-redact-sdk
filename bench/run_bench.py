#!/usr/bin/env python3
"""Interleaved four-SDK bench driver (bench format v2).

Runs the per-language harnesses (bench_python.py, bench_ts.mjs,
rust/examples/bench.rs, bench/swift) round-by-round, rotating the
language order every round so machine drift is spread across
languages instead of systematically favoring whichever ran first. Each harness invocation
executes one measured pass in-process (raw samples are reported);
this driver is the single aggregator: pooled median + nearest-rank
p95 over all samples, plus per-round medians.

The driver also enforces the apples-to-apples audit: every harness
reports a SHA-256 digest of the inputs it actually read and of the
outputs it produced (canonical payload format, see
bench_python.py). A mode only counts as compared when all
languages' input digests are equal and their output digests match.

    python3 bench/run_bench.py            # full sweep, writes results.json
    NEURAL=0 python3 bench/run_bench.py   # skip neural modes
    python3 bench/run_bench.py --lengths  # document-length curve,
                                          # writes length_results.json

The --lengths sweep reuses this driver (same harnesses, same
interleaving, same digest gate) over the length-tier corpora from
bench/gen_length_tiers.py, passed to the harnesses via the
BENCH_NOTES_PATH override each harness honors. The canonical sweep
above is untouched by it.
"""
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
HOME = os.path.expanduser("~")
VENV_PY = os.environ.get(
    "VENV_PY", os.path.join(HOME, "workspace/phi-redact/.venv/bin/python"))
MODEL = os.environ.get(
    "MODEL", os.path.join(HOME, "workspace/phi-redact/work/v9/phi-redact-v9-int8.onnx"))
TOK = os.environ.get(
    "TOK", os.path.join(HOME, "workspace/phi-redact/work/v9/mac-model-dir/tokenizer.json"))
TAGS = os.environ.get(
    "TAGS", os.path.join(HOME, "workspace/phi-redact/work/v9/mac-model-dir/tags.json"))
RS_BENCH = os.path.join(ROOT, "rust", "target", "release", "examples", "bench")
SWIFT_BIN = os.environ.get(
    "SWIFT_BIN", os.path.join(
        HOME, "workspace/tooling/swift-6.1/toolchain/usr/bin/swift"))
SW_BENCH_SCRATCH = os.environ.get("SW_BENCH_SCRATCH", "/tmp/phi-bench-swift")
NEURAL = os.environ.get("NEURAL", "1") == "1"

PER_NOTE_MODES = ["scan", "rules", "features"]
BATCH_MODES = ["batch", "batch-features"]
NOTE_ROUNDS = 7
BATCH_ROUNDS = 5

LANGS = ["python", "typescript", "rust", "swift"]
NEURAL_LANGS = ["python", "rust"]


def swift_bench():
    import glob
    if os.environ.get("SW_BENCH"):
        return os.environ["SW_BENCH"]
    hits = sorted(glob.glob(os.path.join(
        SW_BENCH_SCRATCH, "*/release/bench-swift")))
    return hits[0] if hits else None


def ort_dylib():
    import glob
    hits = glob.glob(os.path.join(
        os.path.dirname(os.path.dirname(VENV_PY)), "lib", "python*",
        "site-packages", "onnxruntime", "capi", "libonnxruntime.so.*"))
    return hits[0] if hits else None


def run_harness(lang, mode, warmups, reps, extra=(), notes_path=None):
    env = dict(os.environ)
    env["BENCH_WARMUPS"] = str(warmups)
    env["BENCH_REPS"] = str(reps)
    if notes_path:
        env["BENCH_NOTES_PATH"] = notes_path
    if lang == "python":
        env["PYTHONPATH"] = os.path.join(ROOT, "python")
        cmd = [VENV_PY, os.path.join(HERE, "bench_python.py"), mode, *extra]
        cwd = ROOT
    elif lang == "typescript":
        cmd = ["node", os.path.join(HERE, "bench_ts.mjs"), mode, *extra]
        cwd = ROOT
    elif lang == "swift":
        binary = swift_bench()
        if not binary:
            raise RuntimeError(
                "swift bench binary not found under "
                f"{SW_BENCH_SCRATCH}; build it first (bench/run_bench.sh "
                "does) or set SW_BENCH to the binary path")
        env["BENCH_SDK_ROOT"] = ROOT
        cmd = [binary, mode, *extra]
        cwd = ROOT
    else:
        dylib = ort_dylib()
        if dylib:
            env["ORT_DYLIB_PATH"] = dylib
        cmd = [RS_BENCH, mode, *extra]
        cwd = os.path.join(ROOT, "rust")
    proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True,
                          text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{lang}/{mode} failed ({proc.returncode}):\n{proc.stderr}")
    line = [ln for ln in proc.stdout.splitlines() if ln.strip()][-1]
    return json.loads(line)


def rotate(langs, round_idx):
    k = round_idx % len(langs)
    return langs[k:] + langs[:k]


def median(xs):
    return statistics.median(xs)


def p95(xs):
    s = sorted(xs)
    return s[max(0, math.ceil(0.95 * len(s)) - 1)]


def agg(samples):
    return {"median_ms": round(median(samples), 4),
            "p95_ms": round(p95(samples), 4),
            "samples": len(samples)}


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_out(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


def cpu_name():
    with open("/proc/cpuinfo") as f:
        for line in f:
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return ""


def environment_block():
    return {
        "python": cmd_out([VENV_PY, "--version"]),
        "node": cmd_out(["node", "--version"]),
        "rustc": cmd_out(["rustc", "--version"]),
        "swift": cmd_out([SWIFT_BIN, "--version"]).splitlines()[0]
                 if os.path.exists(SWIFT_BIN) else "",
        "cpu": cpu_name(),
        "cpus": os.cpu_count(),
        "os": " ".join(os.uname()[:3:2]),
        "onnxruntime": "1.30 (python wheel; rust loads the same "
                       "wheel's libonnxruntime.so via ort load-dynamic)",
        "model": "phi-redact-v9-int8.onnx (v9 int8, 11.3 MB)",
    }


def main() -> int:
    # Corpus audit: checked-in batch corpus must equal a fresh
    # gen_corpus.py run (seeded, deterministic).
    gen = subprocess.run([sys.executable, os.path.join(HERE, "gen_corpus.py")],
                         capture_output=True)
    regen_sha = hashlib.sha256(gen.stdout).hexdigest()
    corpus_sha = file_sha256(os.path.join(HERE, "corpus.jsonl"))
    notes_sha = file_sha256(os.path.join(ROOT, "e2e", "parity", "notes.jsonl"))

    digests = {}     # mode -> {lang: {"digest": set, "digest_noscore": set, "input_digest": set}}
    runs = []        # v1-shaped aggregate rows
    per_rep = {}     # mode -> lang -> [round medians]
    pooled = {}      # (mode, lang) -> [samples]
    batch_totals = {}  # (mode, lang) -> [totals_s]
    stages = {}      # lang -> {"tokenize": [], "inference": [], "post": [], "total": []}
    load_s = {}      # lang -> [model_load_s]

    def note_digest(mode, lang, out):
        d = digests.setdefault(mode, {}).setdefault(
            lang, {"digest": set(), "digest_noscore": set(),
                   "input_digest": set()})
        d["digest"].add(out["digest"])
        d["digest_noscore"].add(out["digest_noscore"])
        d["input_digest"].add(out["input_digest"])

    # -- per-note modes, interleaved --------------------------------
    for mode in PER_NOTE_MODES:
        for rnd in range(NOTE_ROUNDS):
            for lang in rotate(LANGS, rnd):
                out = run_harness(lang, mode,
                                  warmups=2 if rnd == 0 else 0, reps=1)
                key = (mode, lang)
                pooled.setdefault(key, []).extend(out["samples_ms"])
                per_rep.setdefault(mode, {}).setdefault(lang, []).append(
                    round(median(out["samples_ms"]), 4))
                note_digest(mode, lang, out)

    # -- batch modes, interleaved -----------------------------------
    for mode in BATCH_MODES:
        for rnd in range(BATCH_ROUNDS):
            for lang in rotate(LANGS, rnd):
                out = run_harness(lang, mode,
                                  warmups=1 if rnd == 0 else 0, reps=1)
                batch_totals.setdefault((mode, lang), []).extend(
                    out["totals_s"])
                per_rep.setdefault(mode, {}).setdefault(lang, []).append(
                    round(out["totals_s"][0], 4))
                note_digest(mode, lang, out)

    # -- neural modes (python + rust only), interleaved -------------
    if NEURAL:
        import tempfile
        assets = tempfile.mkdtemp(prefix="bench-assets-")
        config = os.path.join(assets, "config.json")
        tags = json.load(open(TAGS))
        with open(config, "w") as f:
            json.dump({"id2label": {str(i): t for i, t in enumerate(tags)},
                       "label2id": {t: i for i, t in enumerate(tags)}}, f)
        for mode in ("neural", "neural-stages"):
            for rnd in range(NOTE_ROUNDS):
                for lang in rotate(NEURAL_LANGS, rnd):
                    labels = config if lang == "python" else TAGS
                    out = run_harness(lang, mode,
                                      warmups=2 if rnd == 0 else 0, reps=1,
                                      extra=(MODEL, TOK, labels))
                    load_s.setdefault(lang, []).append(out["model_load_s"])
                    note_digest(mode, lang, out)
                    if mode == "neural":
                        pooled.setdefault((mode, lang), []).extend(
                            out["samples_ms"])
                        per_rep.setdefault(mode, {}).setdefault(lang, []).append(
                            round(median(out["samples_ms"]), 4))
                    else:
                        st = stages.setdefault(lang, {
                            "tokenize": [], "inference": [], "post": [],
                            "total": []})
                        st["tokenize"].extend(out["stages_ms"]["tokenize"])
                        st["inference"].extend(out["stages_ms"]["inference"])
                        st["post"].extend(out["stages_ms"]["post"])
                        st["total"].extend(out["samples_ms"])
                        per_rep.setdefault(mode, {}).setdefault(lang, []).append(
                            round(median(out["samples_ms"]), 4))

    # -- aggregates ---------------------------------------------------
    for (mode, lang), samples in sorted(pooled.items()):
        runs.append({"language": lang, "mode": mode, **agg(samples),
                     "notes": 15})
    for (mode, lang), totals in sorted(batch_totals.items()):
        med = median(totals)
        runs.append({"language": lang, "mode": mode, "docs": 1000,
                     "median_total_s": round(med, 4),
                     "notes_per_sec": round(1000 / med, 1),
                     "per_note_ms": round(med, 4),
                     "reps": len(totals)})
    order = {m: i for i, m in enumerate(
        ["scan", "rules", "features", "batch", "batch-features", "neural"])}
    runs.sort(key=lambda r: (order.get(r["mode"], 9), r["language"]))

    digest_report = {}
    for mode, by_lang in sorted(digests.items()):
        entry = {}
        for lang, d in sorted(by_lang.items()):
            entry[lang] = {
                "digest": sorted(d["digest"]),
                "digest_noscore": sorted(d["digest_noscore"]),
                "input_digest": sorted(d["input_digest"]),
            }
        all_d = {x for d in by_lang.values() for x in d["digest"]}
        all_ns = {x for d in by_lang.values() for x in d["digest_noscore"]}
        all_in = {x for d in by_lang.values() for x in d["input_digest"]}
        entry["match"] = len(all_d) == 1
        entry["match_noscore"] = len(all_ns) == 1
        entry["input_match"] = len(all_in) == 1
        digest_report[mode] = entry

    results = {
        "format": 2,
        "environment": environment_block(),
        "corpus": {
            "notes": {"path": "e2e/parity/notes.jsonl", "count": 15,
                      "file_sha256": notes_sha},
            "batch": {"path": "bench/corpus.jsonl", "count": 1000,
                      "file_sha256": corpus_sha,
                      "regen_sha256": regen_sha,
                      "regen_matches_file": regen_sha == corpus_sha},
        },
        "method": {
            "per_note": f"{NOTE_ROUNDS} rounds x 15 notes, languages "
                        "interleaved with rotating order per round; "
                        "2 warm-up passes in round 0 only; raw samples "
                        "pooled and aggregated centrally by this driver "
                        "(median + nearest-rank p95)",
            "batch": f"{BATCH_ROUNDS} rounds x 1,000 docs, interleaved; "
                     "1 warm-up pass in round 0 only; median of round "
                     "totals (per-round totals preserved in per_rep)",
            "neural_stages": "stage buckets: tokenize = all tokenizer "
                             "work incl. split checks; inference = ONNX "
                             "session.run only; post = the remainder of "
                             "predict (feed/tensor assembly, softmax/"
                             "argmax, BIOES decode, dedupe). Python "
                             "measured by timing wrappers swapped onto "
                             "the real OnnxEngine; Rust measured inside "
                             "OnnxEngine::predict_timed",
        },
        "digests": digest_report,
        "runs": runs,
        "neural_stages": {
            lang: {stage: agg(vals) for stage, vals in st.items()}
            for lang, st in stages.items()
        },
        "model_load_s": {lang: round(median(v), 3)
                         for lang, v in load_s.items()},
        "per_rep": per_rep,
    }
    out_path = os.path.join(HERE, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out_path}")
    bad = [m for m, d in digest_report.items()
           if not (d["match"] and d["input_match"])]
    if bad:
        print(f"DIGEST MISMATCH in modes: {bad}", file=sys.stderr)
        return 1
    print("all output digests match across languages")
    return 0


LENGTH_TIERS = [500, 2000, 10000, 50000]
LENGTH_ROUNDS = 3
# Neural passes at length are seconds per document, not the fixture
# notes' ~1.5 ms: rounds are sized per tier so the sweep stays
# tractable (documents are the samples; even 1 round is 20-100).
LENGTH_NEURAL_ROUNDS = {500: 2, 2000: 2, 10000: 1, 50000: 1}
LENGTH_STAGES_ROUNDS = {500: 2, 2000: 2, 10000: 1, 50000: 1}


def neural_config_path():
    import tempfile
    assets = tempfile.mkdtemp(prefix="bench-assets-")
    config = os.path.join(assets, "config.json")
    tags = json.load(open(TAGS))
    with open(config, "w") as f:
        json.dump({"id2label": {str(i): t for i, t in enumerate(tags)},
                   "label2id": {t: i for i, t in enumerate(tags)}}, f)
    return config


def _digest_report(digests):
    report = {}
    for mode, by_lang in sorted(digests.items()):
        entry = {}
        for lang, d in sorted(by_lang.items()):
            entry[lang] = {
                "digest": sorted(d["digest"]),
                "digest_noscore": sorted(d["digest_noscore"]),
                "input_digest": sorted(d["input_digest"]),
            }
        all_d = {x for d in by_lang.values() for x in d["digest"]}
        all_ns = {x for d in by_lang.values() for x in d["digest_noscore"]}
        all_in = {x for d in by_lang.values() for x in d["input_digest"]}
        entry["match"] = len(all_d) == 1
        entry["match_noscore"] = len(all_ns) == 1
        entry["input_match"] = len(all_in) == 1
        report[mode] = entry
    return report


def _length_skeleton():
    return {
        "format": 1,
        "kind": "length_curve",
        "environment": environment_block(),
        "method": {
            "tiers": "bench/gen_length_tiers.py (seeded, deterministic; "
                     "each tier file regen-checked against a fresh "
                     "generator run, SHA-256)",
            "per_note": f"{LENGTH_ROUNDS} rounds x tier docs, "
                        "languages interleaved with rotating order "
                        "per round; 2 warm-up passes in round 0 only; "
                        "raw samples pooled and aggregated centrally "
                        "(median + nearest-rank p95)",
            "neural": "rounds per tier sized to the per-document "
                      "cost (neural predict is seconds per document "
                      "at the top tiers): neural "
                      f"{LENGTH_NEURAL_ROUNDS}, neural-stages "
                      f"{LENGTH_STAGES_ROUNDS} (python + rust); same "
                      "interleaving and stage buckets as the "
                      "canonical sweep",
            "digest_gate": "identical to the canonical sweep: a mode "
                           "at a tier counts as compared only when "
                           "all languages' input digests are equal "
                           "and their output digests match",
        },
        "tiers": {},
    }


def _write_length_results(results):
    out_path = os.path.join(HERE, "length_results.json")
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp, out_path)
    return out_path


def run_lengths() -> int:
    """Document-length curve: the canonical per-note modes (scan,
    rules, features) plus neural/neural-stages, run over the
    length-tier corpora from bench/gen_length_tiers.py under the
    same interleaving and digest gate as the canonical sweep.
    Writes bench/length_results.json (format 1); results.json is
    not touched.

    Results are written incrementally (atomically, after each
    tier's local section and again after its neural section) and
    merged with any tiers already present, so a killed run loses
    at most the section in flight. `--tier N` restricts the run to
    one tier; `--neural-only` recomputes just a tier's neural
    sections; LENGTH_NEURAL_MODES=neural|neural-stages restricts
    the neural modes run."""
    args = sys.argv[1:]
    only_tier = (int(args[args.index("--tier") + 1])
                 if "--tier" in args else None)
    neural_only = "--neural-only" in args
    modes_env = os.environ.get("LENGTH_NEURAL_MODES", "both")
    neural_modes = [("neural", LENGTH_NEURAL_ROUNDS),
                    ("neural-stages", LENGTH_STAGES_ROUNDS)]
    if modes_env == "neural":
        neural_modes = neural_modes[:1]
    elif modes_env == "neural-stages":
        neural_modes = neural_modes[1:]

    out_path = os.path.join(HERE, "length_results.json")
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)
    else:
        results = _length_skeleton()
    tiers_out = results.setdefault("tiers", {})

    for tier in ([only_tier] if only_tier else LENGTH_TIERS):
        path = os.path.join(HERE, "lengths", f"tier_{tier}.jsonl")
        # Corpus audit: checked-in tier file must equal a fresh
        # gen_length_tiers.py run for that tier (seeded).
        gen = subprocess.run(
            [sys.executable,
             os.path.join(HERE, "gen_length_tiers.py"), str(tier)],
            capture_output=True)
        regen_sha = hashlib.sha256(gen.stdout).hexdigest()
        tier_sha = file_sha256(path)
        with open(path) as f:
            docs = [json.loads(ln)["note"] for ln in f if ln.strip()]
        lens = sorted(len(d) for d in docs)
        median_chars = lens[len(lens) // 2]
        corpus = {
            "path": f"bench/lengths/tier_{tier}.jsonl",
            "target_chars": tier,
            "count": len(docs),
            "chars": {"min": lens[0], "median": median_chars,
                      "mean": round(sum(lens) / len(lens)),
                      "max": lens[-1]},
            "file_sha256": tier_sha,
            "regen_sha256": regen_sha,
            "regen_matches_file": regen_sha == tier_sha,
        }

        old = tiers_out.get(str(tier), {})
        digests = {}
        pooled = {}
        per_rep = {}
        stages = {}
        load_s = {}

        def note_digest(mode, lang, out):
            d = digests.setdefault(mode, {}).setdefault(
                lang, {"digest": set(), "digest_noscore": set(),
                       "input_digest": set()})
            d["digest"].add(out["digest"])
            d["digest_noscore"].add(out["digest_noscore"])
            d["input_digest"].add(out["input_digest"])

        def build_entry():
            fresh = {}
            for (mode, lang), samples in sorted(pooled.items()):
                a = agg(samples)
                fresh[(mode, lang)] = {
                    "language": lang, "mode": mode, "docs": len(docs),
                    "median_chars": median_chars, **a,
                    "ms_per_1000_chars": round(
                        a["median_ms"] / median_chars * 1000, 4),
                    "chars_per_sec": round(
                        median_chars / a["median_ms"] * 1000, 1),
                }
            merged = {(r["mode"], r["language"]): r
                      for r in old.get("runs", [])}
            merged.update(fresh)
            order = {m: i for i, m in enumerate(
                ["scan", "rules", "features", "neural"])}
            runs = sorted(merged.values(),
                          key=lambda r: (order.get(r["mode"], 9),
                                         r["language"]))
            merged_digests = dict(old.get("digests", {}))
            merged_digests.update(_digest_report(digests))
            merged_per_rep = dict(old.get("per_rep", {}))
            merged_per_rep.update(per_rep)
            return {
                "corpus": corpus,
                "runs": runs,
                "digests": merged_digests,
                "neural_stages": (
                    {lang: {stage: agg(vals)
                            for stage, vals in st.items()}
                     for lang, st in stages.items()}
                    if stages else old.get("neural_stages", {})),
                "model_load_s": (
                    {lang: round(median(v), 3)
                     for lang, v in load_s.items()}
                    if load_s else old.get("model_load_s", {})),
                "per_rep": merged_per_rep,
            }

        if not neural_only:
            for mode in PER_NOTE_MODES:
                for rnd in range(LENGTH_ROUNDS):
                    for lang in rotate(LANGS, rnd):
                        out = run_harness(lang, mode,
                                          warmups=2 if rnd == 0 else 0,
                                          reps=1, notes_path=path)
                        key = (mode, lang)
                        pooled.setdefault(key, []).extend(
                            out["samples_ms"])
                        per_rep.setdefault(mode, {}).setdefault(
                            lang, []).append(
                            round(median(out["samples_ms"]), 4))
                        note_digest(mode, lang, out)
            tiers_out[str(tier)] = build_entry()
            _write_length_results(results)
            print(f"tier {tier}: local section written", flush=True)

        if NEURAL:
            config = neural_config_path()
            for mode, rounds_map in neural_modes:
                for rnd in range(rounds_map[tier]):
                    for lang in rotate(NEURAL_LANGS, rnd):
                        labels = config if lang == "python" else TAGS
                        out = run_harness(lang, mode,
                                          warmups=2 if rnd == 0 else 0,
                                          reps=1,
                                          extra=(MODEL, TOK, labels),
                                          notes_path=path)
                        load_s.setdefault(lang, []).append(
                            out["model_load_s"])
                        note_digest(mode, lang, out)
                        if mode == "neural":
                            pooled.setdefault((mode, lang), []).extend(
                                out["samples_ms"])
                            per_rep.setdefault(mode, {}).setdefault(
                                lang, []).append(
                                round(median(out["samples_ms"]), 4))
                        else:
                            st = stages.setdefault(lang, {
                                "tokenize": [], "inference": [],
                                "post": [], "total": []})
                            st["tokenize"].extend(
                                out["stages_ms"]["tokenize"])
                            st["inference"].extend(
                                out["stages_ms"]["inference"])
                            st["post"].extend(out["stages_ms"]["post"])
                            st["total"].extend(out["samples_ms"])
                            per_rep.setdefault(mode, {}).setdefault(
                                lang, []).append(
                                round(median(out["samples_ms"]), 4))

        tiers_out[str(tier)] = build_entry()
        _write_length_results(results)
        print(f"tier {tier}: {len(docs)} docs, median "
              f"{median_chars} chars — done", flush=True)

    any_mismatch = []
    for t, entry in tiers_out.items():
        for mode, d in entry.get("digests", {}).items():
            if not (d["match"] and d["input_match"]):
                any_mismatch.append(f"tier {t}/{mode}")
    print(f"wrote {out_path}")
    if any_mismatch:
        print(f"DIGEST MISMATCH in: {any_mismatch}", file=sys.stderr)
        return 1
    print("all output digests match across languages at every tier")
    return 0


if __name__ == "__main__":
    if "--lengths" in sys.argv:
        raise SystemExit(run_lengths())
    raise SystemExit(main())
