# Performance comparison — Python vs TypeScript vs Rust SDKs

Measured 2026-10-27 in this sandbox, all three SDKs on the same
machine, same corpora, same work per mode. Raw output:
[`perf-results.json`](perf-results.json) (regenerate with
`bench/run_bench.sh` from the repo root; harnesses in `bench/` and
`rust/examples/bench.rs`).

## Results

Per-note latency over the 15 e2e fixture notes (median / p95, ms):

| Mode | Python | TypeScript | Rust |
| --- | ---: | ---: | ---: |
| (a) Rules-only pipeline | 0.0538 / 0.1547 | 0.0275 / 0.0616 | **0.0071 / 0.0191** |
| (b) Full feature pipeline | 0.0711 / 0.1740 | 0.0417 / 0.1397 | **0.0127 / 0.0398** |
| (d) Full neural pipeline (v9 int8) | 1.5256 / 3.1132 | n/a | 1.6330 / 4.1064 |
| Neural model load (s, once) | 0.728 | n/a | 0.327 |

(c) Batch throughput, `redact_batch` over 1,000 documents:

| | Python | TypeScript | Rust |
| --- | ---: | ---: | ---: |
| notes/sec | 13,708 | 45,407 | **98,530** |
| amortized ms/note | 0.0729 | 0.0220 | 0.0101 |

Read: on the local (non-neural) pipeline Rust is ~7.6× Python and
~3.9× TypeScript at the median for rules-only, ~5.6× / ~3.3× with the
full feature stack on, and ~7.2× / ~2.2× on batch throughput. The
neural rows are effectively tied: both runtimes spend their time
inside the same ONNX Runtime, so the host language contributes
little; Rust loads the model about 2× faster in this harness.

## Method

- **Corpora.** (a), (b), (d): the 15 synthetic fixture notes from
  `e2e/parity/notes.jsonl` (short notes, ~60–200 chars). (c): 1,000
  documents built by `bench/gen_corpus.py` — fixture notes repeated
  with a varying `Log entry N.` suffix in a seeded shuffle.
  **Fixture-derived, not representative clinical text**; longer
  documents shift absolute numbers (chunking, more spans) though the
  local pipeline is linear in text length.
- **Work per mode.** (a) default `Redactor` redaction. (b) one fixed
  configuration in all three languages: hash operator (salt
  `"bench"`), denylist `["Maria"]`, two recognizers (EMPLOYEE_ID
  `EMP\d{6}` @ 0.9; REF_CODE `[A-Z]{2}-\d{4}` @ 0.8), per-label
  threshold REF_CODE 0.9, review margin 0.2. (c) `redact_batch` /
  `redactBatch` with the default configuration. (d) `Redactor` with
  the real **v9 int8 ONNX** (11.3 MB) + v9 tokenizer; model load timed
  separately and excluded from per-note numbers.
- **Timing.** Per-note modes: each note timed individually
  (`perf_counter_ns` / `hrtime.bigint` / `Instant`), 2 warm-up passes
  discarded, 7 measured passes, all 105 samples pooled; median and
  nearest-rank p95 reported. Batch: 1 warm-up, 5 timed reps, median
  total. Sequential runs on an otherwise idle VM.
- **Builds.** Rust: `cargo build --release` (opt-level 3). Node: the
  built `typescript/dist`. Python: the source tree on PYTHONPATH
  (the venv's installed copy is stale), CPython as-is.
- **Environment.** Sandbox VM, 2 CPUs of an AMD EPYC 9D25, Linux
  7.0.0-39-generic; Python 3.12.3, Node v24.22.1, rustc 1.99.1.
  Python/Rust neural both ran ONNX Runtime 1.30 (Python: its bundled
  wheel; Rust: the same wheel's `libonnxruntime.so` via `ort`'s
  `load-dynamic` — see the README's neural section for why).

## Caveats

- **TypeScript neural is n/a, by design**: the TS SDK has no local
  neural engine (it ships a `RemoteRedactor` HTTP client instead), so
  there is no honest TS number for mode (d). TS rows are rules-only
  local pipeline, like (a)/(b).
- Sandbox VM numbers are noisy at the microsecond scale (repeat
  sweeps moved batch throughput ±15%); ratios between languages were
  stable across sweeps, absolute values less so. Do not quote these
  as production benchmarks.
- The fixture notes are short. On the 500-note clinical holdout used
  in the Presidio comparison, Python's full neural pipeline measured
  8.4 ms/note p50 — document length dominates.
- One Rust-specific finding shaped these numbers: `regex::Regex::new`
  costs ~0.2 ms per pattern (Unicode tables). The Python SDK
  recompiles recognizer/denylist patterns per call but pays once via
  CPython's `re` cache. The Rust port therefore caches compiled
  patterns process-globally (see `rust/README.md`); without the
  cache, Rust's feature-pipeline median was 0.50 ms/note instead of
  0.0127.
