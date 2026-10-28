# Performance — Python vs TypeScript vs Rust SDKs, plus Swift (PhiRedactKit)

Re-measured 2026-10-27 with an audited, stage-decomposed harness
(bench format v2), superseding the same-day first-pass numbers in
git history. The audit found three defects in the original sweep —
the "rules-only" mode actually ran the full default redact path,
outputs were never compared across languages, and languages ran in
a fixed sequential order — and fixed all three before re-measuring
(see "Apples-to-apples verification" below). A fourth
implementation, the Swift **PhiRedactKit** that powers the Mac
app, was added later the same day under the identical protocol.
That first Swift sweep found the kit compiling its recognizer
and denylist regexes on every call; the kit has since gained a
compiled-pattern cache (Mac repo PR #7), and the whole sweep was
re-run once more as one coherent pass — **every figure below is
from that post-fix run** (the other three implementations' medians
moved ≤ ~9% from the audit run, within the noise documented
under Caveats; the pre-fix Swift figures are quoted in "What
this means for the Mac app" where the improvement is measured).
Headline: on the local pipeline Rust is ~7×
Python and ~4× TypeScript at the median; Swift tracks Python on
the plain pipeline and, after the cache fix, sits ~1.6× behind
Python on the feature pipeline (see "What this means for the
Mac app"). The
neural (RoBERTa-style token classifier) pipeline is dominated
(~85–90%) by ONNX Runtime inference in both SDK languages that
have it, so Python and Rust land within noise of each other there.
Raw per-rep data: [`bench/results.json`](bench/results.json);
reproduce with `bench/run_bench.sh`. Extended notes:
[`rust/PERF.md`](rust/PERF.md).

## Results

Per-note latency over the 15 e2e fixture notes (median / p95, ms;
105 pooled samples per cell):

| Stage | Python | TypeScript | Rust | Swift (kit) |
| --- | ---: | ---: | ---: | ---: |
| (a) Rules scan only (patterns + validators) | 0.0477 / 0.1367 | 0.0285 / 1.0775 | **0.0069 / 0.0236** | 0.0438 / 0.1157 |
| (b) Local pipeline (scan + merge + operators) | 0.0538 / 0.1443 | 0.0356 / 0.0745 | **0.0080 / 0.0231** | 0.0495 / 0.1280 |
| (c) Full feature pipeline ((b) + recognizers, lists, thresholds/review) | 0.0739 / 0.1840 | 0.0583 / 0.1819 | **0.0130 / 0.0450** | 0.1145 / 0.5559 |
| (e) Rules + neural pipeline (v9 int8, combined) | 1.6797 / 4.5311 | n/a | 1.5809 / 5.6550 | n/a |
| Neural model load (s, once) | 0.303 | n/a | 0.176 | n/a |

Batch throughput, `redact_batch` over 1,000 documents (notes/sec;
amortized ms/note in parentheses):

| Config | Python | TypeScript | Rust | Swift (kit) |
| --- | ---: | ---: | ---: | ---: |
| (b) Local pipeline | 13,263 (0.0754) | 51,814 (0.0193) | **97,087 (0.0103)** | 15,723 (0.0636) |
| (c) Full feature pipeline | 10,684 (0.0936) | 23,148 (0.0432) | **57,804 (0.0173)** | 5,280 (0.1894) |

Swift neural is n/a here, not zero: PhiRedactKit's neural engine
is Core ML, which exists only on Apple platforms, and this bench
runs on Linux. The kit's local stages (a)–(c) are platform-portable
Foundation work and are measured like the others, digest-verified
against them (see below).

Neural stage decomposition — `predict()` split into buckets
(median ms per note; both languages bind the same ONNX Runtime
1.30 native library):

| Stage | Python | Rust |
| --- | ---: | ---: |
| Tokenize (tokenizer encode incl. split checks) | 0.1104 | 0.1002 |
| ONNX inference (`session.run` only) | 1.2087 | 1.4502 |
| Post (feed/tensor assembly, softmax/argmax, BIOES decode, dedupe) | 0.1112 | 0.0444 |
| `predict()` total | 1.4172 | 1.6340 |

**Rules vs neural, in one line:** the rules engine costs
0.05–0.07 ms/note in Python, ~0.05 ms in Swift and ~0.01 ms/note
in Rust; adding the neural layer costs a further ~1.6 ms/note in
both SDK languages that run it locally — roughly 85–89% of
`predict()` is the ONNX `session.run` itself, ~0.1 ms is
tokenization, and the only stage where the host language is really
visible is post-processing (Python 0.111 ms vs Rust 0.044 ms).
(The stage totals and the combined row above come from separate
interleaved round sets and differ by ~10% run-to-run variance;
the stage *shares* are the reliable output of the decomposition.)

## What this means for the Mac app

The Swift column is PhiRedactKit — the Mac app's actual engine —
measured here on Linux Swift 6.1 (see the caveat below). Read
against the Rust column, it answers the recurring "should the app
use a Rust backend?" question with numbers:

- On the plain local pipeline (b) the kit runs 0.0495 ms/note —
  essentially Python's speed (0.0538), ~1.4× TypeScript, ~6.2×
  slower than Rust. The feature pipeline (c) is where the first
  Swift sweep found a real defect: the kit compiled its
  recognizer and denylist `NSRegularExpression`s on every call,
  with no compiled-pattern cache — the same trap the Rust port
  measured at 0.50 ms/note before its process-global cache —
  and stage (c) measured 0.1522 ms/note (~11.8× Rust, ~2×
  Python). The kit now routes both call sites through a
  process-global `RegexCache` (Mac repo PR #7; only successful
  compiles are cached, so invalid patterns throw exactly as
  before), and this run measures stage (c) at 0.1145 ms/note —
  a ~25% improvement (batch (c): 4,688 → 5,280 notes/sec).
  A residual gap to Rust (~8.8×) and Python (~1.6×) at (c)
  remains; it is no longer compilation — every compile site in
  the kit is now cached or compile-once — and is left as
  unattributed host-side overhead rather than explained away.
- What it would buy: against the ~1.6 ms/note the neural
  layer costs on CPU ONNX in this bench, the kit's local pipeline
  is ~3% of the per-note total at (b) and ~7% at (c).
  Swapping the kit's backend for Rust would save at most
  ~0.04 ms/note at (b) and ~0.10 ms/note at (c) — single-digit
  percentages of a redaction, far below any interaction
  threshold. On a real Mac the neural layer runs on Core ML
  (Neural Engine), which this Linux bench cannot measure; even
  in the limit where inference were free, the absolute saving
  stays ~0.04–0.10 ms per note.

Plainly: on these numbers there is no end-to-end performance case
for a Rust backend in the Mac app. The kit's one real deficit —
per-call regex compilation in the feature path — was a caching
fix in Swift, not a rewrite, and it has shipped.

## Apples-to-apples verification

The v2 harness was built to make the comparison provable rather
than plausible:

- **Corpus identity.** All runners read the same two files:
  `e2e/parity/notes.jsonl` (SHA-256
  `00171a44500d19421a0fcd8a567d91dbfa9d043f93cd4fd1f0e8cc7484874a73`,
  15 notes, pure ASCII, 0–130 chars) and `bench/corpus.jsonl`
  (SHA-256 `c7ec2f1101f2d52552d0994d8b38250e6bf7ee666c5e5c61eda3a8ca2bc8b41e`,
  1,000 docs). The batch corpus is generated by
  `bench/gen_corpus.py` from a fixed seed; the driver re-generates
  it and confirms the hash matches the file on disk. Each runner
  additionally reports a digest of the parsed inputs it actually
  read — identical across languages in every mode.
- **Config identity.** The (c) configuration is one fixed literal
  config written out in each harness: hash operator (salt
  `"bench"`), denylist `["Maria"]`, recognizers EMPLOYEE_ID
  `EMP\d{6}` @ 0.9 and REF_CODE `[A-Z]{2}-\d{4}` @ 0.8, per-label
  threshold REF_CODE 0.9, review margin 0.2. Modes (a)/(b) use each
  SDK's default configuration.
- **Output-digest equality.** Every runner hashes a canonical
  serialization of its outputs (redacted text, entity spans with
  label/score/source/placeholder, review candidates) for every
  mode. In this run **all digests matched across all four
  implementations in the five local modes** (scan, rules,
  features, batch, batch-features), and across Python and Rust in
  the neural modes — including the scored digest for the full
  neural pipeline, i.e. Python and Rust produced identical spans
  with identical scores (to 4 decimals) on the real v9 model, and
  identical batch outputs over all 1,000 documents. A mode only
  counts as compared when its digests match; a mismatch fails the
  driver.
- **Timed regions.** Engine construction, file I/O, and model load
  are outside every timed region in all four harnesses; model load
  is reported separately. Each invocation runs an untimed digest
  pass before timing, so every language is measured in steady
  state; round 0 adds 2 explicit warm-up passes.
- **Interleaving + central aggregation.** The driver
  (`bench/run_bench.py`) runs 7 rounds (5 for batch), rotating the
  language order each round, and pools the raw samples itself —
  one median/p95 implementation for all languages, per-round
  medians preserved in `bench/results.json`.
- **Inherent platform asymmetries (facts, not fixes).** TypeScript
  constructs a fresh `RegExp` for every pattern on every scan
  (required: `/g` regexes are stateful) — that cost is part of its
  steady state. Python "recompiles" recognizer/denylist patterns
  per call but pays once via CPython's `re` cache. Rust compiles
  recognizer/denylist patterns through a process-global cache and
  its built-in rule patterns once; without the cache its feature
  pipeline measured 0.50 ms/note in the first-pass sweep. Swift's
  kit compiles its built-in rules once, and — since the cache fix
  in Mac repo PR #7 — routes recognizer and denylist patterns
  through a process-global cache like Rust's; the first Swift
  sweep, run before that fix, measured the uncached cost directly
  (stage (c) 0.1522 ms/note). Rust
  public offsets are character indices, TypeScript's and Swift's
  are UTF-16 code units, Python's are code points — identical on this
  all-ASCII corpus, a real difference on non-ASCII text. The
  TypeScript bench runs the prebuilt `typescript/dist` (verified
  current against `src`); Python runs the source tree on
  PYTHONPATH; Rust runs `cargo build --release` with the `neural`
  feature enabled for all modes (the feature only adds the neural
  module; local-pipeline code is identical); Swift runs a
  `swift build -c release` executable that depends on PhiRedactKit
  by path (`bench/swift/`), built with the scratch path outside
  the repo.

## Choosing a language

Grounded in the measurements above plus the SDKs' actual feature
sets:

- **Python** — the fullest surface: the only SDK with the logging
  kit (CloudWatch/Datadog scrubbing), a local neural engine, and
  the reference implementation every port is parity-tested
  against. Its local pipeline is the slowest of the three SDKs
  (13.3k notes/sec batch) and its neural pipeline runs ≈590
  notes/sec single-threaded at the median — ample for application
  backends, log pipelines, and batch jobs that are not
  throughput-bound.
- **TypeScript** — the only SDK that runs in the browser, and the
  natural fit for Node services; local pipeline performance sits
  between Python and Rust (51.8k notes/sec batch). It has **no
  local neural engine by design**: neural detection goes through
  the `redact serve` endpoint (`RemoteRedactor`), so a TS
  deployment that needs the model also runs a server. No logging
  kit (scrub in your logging transport with `scrub()`).
- **Rust** — the throughput choice: 5–8× Python across the local
  stages, 97k notes/sec batch, and a local neural engine
  (feature-gated) whose inference ties Python's because both bind
  the same ONNX Runtime — its edge is the host-side work around
  inference (post-processing ~2.5× faster, model load ~1.7×
  faster in this run). Best for high-volume pipelines, CLIs, and embedding;
  the cost is integration effort (native build, FFI if you embed
  it in another runtime) and no logging kit.

## Environment and method

- Sandbox VM, 2 vCPUs of an AMD EPYC 9D25, Linux 7.0.0-39-generic;
  Python 3.12.3, Node v24.22.1, rustc 1.99.1, Swift 6.1
  (swift-6.1-RELEASE, Linux x86_64 — the same toolchain as the
  kit's Linux CI). One session, one
  machine, interleaved rounds (see above).
- Corpora: the 15 synthetic fixture notes (short, ~0–130 chars)
  and the 1,000-doc fixture-derived batch corpus. **Fixture-derived,
  not representative clinical text**; longer documents shift
  absolute numbers (chunking engages, more spans), though the
  local pipeline is linear in text length.
- Neural: the real **v9 int8 ONNX** (11.3 MB) — a RoBERTa-style
  token classifier — + v9 tokenizer. Python inference runs its
  bundled ONNX Runtime 1.30 wheel; Rust loads the same wheel's
  `libonnxruntime.so` via `ort`'s `load-dynamic`.

## Caveats

- Sandbox VM numbers are noisy at the microsecond scale. Against
  the immediately preceding audit sweep, the Python / TypeScript /
  Rust local-stage medians in this run moved ≤ ~9% cell-for-cell,
  and language ratios were consistent across rounds; absolute
  values are the less stable statistic. TypeScript's scan p95
  (1.08 ms) again reflects a handful of >1 ms outlier samples
  (fresh-process JIT/GC effects) while its round medians sat at
  0.020–0.035 ms; neural p95s remain outlier-driven (4.53 / 5.66
  ms this sweep vs 5.96 / 8.29 ms in the audit sweep) while
  neural medians held within ~8%. Swift's round medians were
  tight for scan (0.0428–0.0443 ms) and, after the cache fix,
  tighter for the feature pipeline (0.113–0.141 ms, vs
  0.133–0.180 ms uncached). Do not quote
  these as production benchmarks.
- The Swift numbers are **Linux** numbers: Swift 6.1 with Linux
  Foundation (`NSRegularExpression` over ICU), release build.
  The Mac app links macOS Foundation and runs the neural layer
  on Core ML / Neural Engine, neither of which exists here — so
  the Swift column measures the kit's algorithms, not the app's
  on-device latency, and its neural row is n/a rather than an
  estimate.
- Model-load times in this run (0.30 s / 0.18 s) benefited from a
  warm page cache after repeated loads in the same session; the
  first-pass sweep measured 0.73 s / 0.33 s. Treat load time as
  "a few hundred ms", not a precise figure.- The fixture notes are short. On the 500-note clinical holdout
  used in the Presidio comparison, Python's full neural pipeline
  measured 8.4 ms/note p50 — document length dominates.
