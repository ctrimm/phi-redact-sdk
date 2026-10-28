# Performance comparison — Python vs TypeScript vs Rust SDKs, plus Swift (PhiRedactKit)

Re-measured 2026-10-27 with the audited bench harness (format
v2), superseding the first-pass numbers from earlier the same day.
A fourth implementation (Swift PhiRedactKit) was added under the
identical protocol, and after the kit gained a compiled-pattern
cache (Mac repo PR #7) the full four-implementation sweep was
re-run once more; **the tables below are that canonical post-fix
sweep**, identical to the repo-root
[`PERFORMANCE.md`](../PERFORMANCE.md), which also carries the
verification section, rules-vs-neural summary, and the
language-choice guide. Raw per-rep
data: [`bench/results.json`](../bench/results.json) (mirrored to
[`perf-results.json`](perf-results.json)); reproduce with
`bench/run_bench.sh` from the repo root.

## Results

Per-note latency over the 15 e2e fixture notes (median / p95, ms;
105 pooled samples per cell):

| Stage | Python | TypeScript | Rust | Swift (kit) |
| --- | ---: | ---: | ---: | ---: |
| (a) Rules scan only | 0.0477 / 0.1367 | 0.0285 / 1.0775 | **0.0069 / 0.0236** | 0.0438 / 0.1157 |
| (b) Local pipeline | 0.0538 / 0.1443 | 0.0356 / 0.0745 | **0.0080 / 0.0231** | 0.0495 / 0.1280 |
| (c) Full feature pipeline | 0.0739 / 0.1840 | 0.0583 / 0.1819 | **0.0130 / 0.0450** | 0.1145 / 0.5559 |
| (e) Rules + neural (v9 int8) | 1.6797 / 4.5311 | n/a | 1.5809 / 5.6550 | n/a |
| Neural model load (s, once) | 0.303 | n/a | 0.176 | n/a |

Batch throughput, `redact_batch` over 1,000 documents:

| Config | Python | TypeScript | Rust | Swift (kit) |
| --- | ---: | ---: | ---: | ---: |
| (b) Local pipeline | 13,263 notes/s (0.0754 ms/note) | 51,814 (0.0193) | **97,087 (0.0103)** | 15,723 (0.0636) |
| (c) Full feature pipeline | 10,684 (0.0936) | 23,148 (0.0432) | **57,804 (0.0173)** | 5,280 (0.1894) |

Neural stage decomposition (`predict()` buckets, median ms/note;
both languages bind ONNX Runtime 1.30):

| Stage | Python | Rust |
| --- | ---: | ---: |
| Tokenize | 0.1104 | 0.1002 |
| ONNX inference (`session.run`) | 1.2087 | 1.4502 |
| Post (assembly, softmax/argmax, decode, dedupe) | 0.1112 | 0.0444 |
| `predict()` total | 1.4172 | 1.6340 |

Read: the local pipeline is 5–8× faster in Rust than Python and
~4–5× faster than TypeScript on the per-note stages (2–3× on
batch). The neural layer
costs a further ~1.6 ms/note in both languages that have it;
~85–89% of `predict()` is the ONNX session call itself, which is why
the neural rows tie — the host language only shows in tokenize
(~0.1 ms) and post-processing (Python 0.111 vs Rust 0.044 ms).
Swift tracks Python on stages (a)/(b); at (c) its first sweep
(uncached recognizer/denylist compilation) measured
0.1522 ms/note, and the pattern cache in Mac repo PR #7 brought
it to 0.1145 — still the slowest feature pipeline of the four,
with the residual gap not attributable to compilation (every
compile site in the kit is now cached or compile-once).

## What the audit changed

The first-pass sweep (same day, earlier) had three defects, found
by reading the harnesses before re-measuring:

1. **Mislabeled mode.** The old "rules-only" row ran the full
   default redact path in all three languages (scan + merge +
   placeholder rendering), not a raw rules scan. Format v2 splits
   (a) raw scan from (b) the local pipeline; the old (a) numbers
   are closest to the new (b) row.
2. **No output verification.** The old harnesses emitted only
   timings. V2 runners hash a canonical serialization of every
   output (redacted text, spans incl. scores, review candidates)
   and the driver refuses to count a mode as compared unless the
   digests match. In this run **all digests matched in all modes
   across all four implementations** (Swift included from its
   first sweep on), including identical neural spans+scores from
   the Python and Rust engines on the real v9 model, and identical
   batch outputs over all 1,000 docs. Corpus identity is pinned
   by file SHA-256 plus a per-runner digest of the parsed inputs,
   and the batch corpus regeneration (seeded) is verified
   byte-identical.
3. **Fixed sequential order.** Languages previously ran
   Python → TypeScript → Rust, once each, so machine drift
   systematically favored one end. V2 interleaves: 7 rounds
   (5 for batch) with rotating language order, raw samples pooled
   and aggregated centrally (`bench/run_bench.py`), per-round
   medians preserved in the raw data.

Two harness bugs were found and fixed along the way: batch
digests were tied to the warm-up pass (so `BENCH_WARMUPS=0`
invocations reported none), and timed passes in later rounds of
a fresh process could include cold pattern-cache costs — the
untimed digest pass now runs before timing in every invocation,
putting every language in steady state (for TypeScript, fresh
`RegExp` construction per scan *is* its steady state; it is
reported as an inherent asymmetry, not warmed away).

## Method

- **Corpora.** Per-note modes: the 15 synthetic fixture notes
  from `e2e/parity/notes.jsonl` (pure ASCII, 0–130 chars). Batch:
  1,000 documents from `bench/gen_corpus.py` — fixture notes
  repeated with a varying `Log entry N.` suffix in a seeded
  shuffle. **Fixture-derived, not representative clinical text.**
- **Work per mode.** (a) `scan` — patterns + validators only.
  (b) default `redact` — scan + merge + operator rendering.
  (c) (b) plus one fixed config in all four implementations: hash
  operator (salt `"bench"`), denylist `["Maria"]`, recognizers
  EMPLOYEE_ID `EMP\d{6}` @ 0.9 and REF_CODE `[A-Z]{2}-\d{4}` @ 0.8,
  REF_CODE threshold 0.9, review margin 0.2. (e) `redact` with the
  real **v9 int8 ONNX** (11.3 MB, a RoBERTa-style token
  classifier) + v9 tokenizer; stage buckets measured inside the
  real predict path (Python: timing wrappers swapped onto
  `OnnxEngine`; Rust: `OnnxEngine::predict_timed`).
- **Timing.** Per-note: 7 interleaved rounds × 15 notes, raw
  samples pooled (105), median + nearest-rank p95. Batch: 5
  interleaved rounds, median of round totals. Engine
  construction, file I/O, and model load are outside all timed
  regions.
- **Builds.** Rust: `cargo build --release --features neural`
  (opt-level 3; the feature only adds the neural module). Node:
  the built `typescript/dist` (verified current vs `src`).
  Python: the source tree on PYTHONPATH, CPython as-is. Swift:
  `swift build -c release` executable depending on PhiRedactKit
  by path (`bench/swift/`), scratch path outside the repo.
- **Environment.** Sandbox VM, 2 CPUs of an AMD EPYC 9D25, Linux
  7.0.0-39-generic; Python 3.12.3, Node v24.22.1, rustc 1.99.1,
  Swift 6.1 (Linux).
  Python/Rust neural both ran ONNX Runtime 1.30 (Python: bundled
  wheel; Rust: the same wheel's `libonnxruntime.so` via `ort`'s
  `load-dynamic`). Swift's neural engine is Core ML (Apple
  platforms only) and is n/a here.

## Caveats

- **TypeScript neural is n/a, by design**: no local neural engine
  (it ships a `RemoteRedactor` HTTP client instead), so there is
  no honest TS number for mode (e). TS rows are the local pipeline.
- Sandbox noise is real at this scale: across the day's sweeps,
  local-stage medians reproduced within ~7–9% (TypeScript scan
  moved 19% between two audit-era runs), TypeScript batch
  throughput varied ~20% between sweeps, and neural p95s are
  outlier-driven while neural medians stayed within ~12%.
  Within this sweep, TypeScript's scan p95 (1.08 ms) is a
  handful of >1 ms outliers in fresh processes, and Swift's
  feature-pipeline round medians ran 0.113–0.141 ms. Medians
  and cross-round ratios are the stable statistics.
- Model-load figures in this run (0.30 s Python / 0.18 s Rust)
  are warm-page-cache values after repeated loads; the first-pass
  sweep measured 0.73 s / 0.33 s.
- The fixture notes are short. On the 500-note clinical holdout
  used in the Presidio comparison, Python's full neural pipeline
  measured 8.4 ms/note p50 — document length dominates.
- One Rust-specific finding shaped these numbers: `regex::Regex::new`
  costs ~0.2 ms per pattern (Unicode tables). Python pays that cost
  once via CPython's `re` cache; the Rust port caches compiled
  patterns process-globally (see `rust/README.md`); without the
  cache, Rust's feature-pipeline median was 0.50 ms/note instead
  of ~0.013.
