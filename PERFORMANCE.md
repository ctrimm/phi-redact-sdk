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

## Document length

Everything above is measured on ~60-character fixture notes.
Real inputs are longer, and cost does not scale uniformly, so the
same harness also ran a length curve: four seeded corpora
(`bench/gen_length_tiers.py`, deterministic; each file
regen-checked by SHA-256) composed from the fixture sentences
plus PHI-bearing and filler clinical/case-note templates —
~500, ~2,000, ~10,000 and ~50,000 characters. Same driver, same
interleaving, same digest gate; the corpora are passed to the
harnesses via a `BENCH_NOTES_PATH` override that defaults to the
canonical files. Local modes ran 3 rounds; neural rounds were
sized per tier (2/2/1/1) because neural predict costs seconds
per document at the top tier. Raw data:
[`bench/length_results.json`](bench/length_results.json);
reproduce with `python3 bench/run_bench.py --lengths`.
(Tier files: 500 = `bb360752…`, 2,000 = `6a628757…`,
10,000 = `15098851…`, 50,000 = `65474993…`.)

What each tier corresponds to in practice:

| Tier (chars) | Actual median | Docs | Representative use case |
| --- | ---: | ---: | --- |
| ~100 | ~62 | 15 + 1,000 | A log line, a chat message, a single form field — the logging-kit and LLM prompt-scrub regime (the headline tables above) |
| ~500 | 500 | 100 | A patient portal message, a referral note, a ticket comment |
| ~2,000 | 1,991 | 100 | A standard encounter or caseworker note; roughly two minutes of dictated speech-to-text |
| ~10,000 | 9,984 | 50 | A full speech-to-text encounter transcript (10+ minutes of dictation), a discharge summary, a 2–3 page document |
| ~50,000 | 49,985 | 20 | A multi-page record bundle (10+ pages) processed as one document — the document-redaction pipeline's batch regime |

Median ms per document (medians over pooled per-document
samples; the ~60 row is the canonical sweep above):

| Tier | Stage | Python | TypeScript | Rust | Swift (kit) |
| --- | --- | ---: | ---: | ---: | ---: |
| ~60 (fixture notes) | (a) Rules scan | 0.0477 | 0.0285 | 0.0069 | 0.0438 |
| ~60 (fixture notes) | (b) Local pipeline | 0.0538 | 0.0356 | 0.0080 | 0.0495 |
| ~60 (fixture notes) | (c) Feature pipeline | 0.0739 | 0.0583 | 0.0130 | 0.1145 |
| ~60 (fixture notes) | (e) Neural | 1.6797 | n/a | 1.5809 | n/a |
| 500 (median 500) | (a) Rules scan | 0.3642 | 0.0360 | 0.0273 | 0.2298 |
| 500 (median 500) | (b) Local pipeline | 0.3647 | 0.0546 | 0.0343 | 0.2563 |
| 500 (median 500) | (c) Feature pipeline | 0.4214 | 0.0966 | 0.0500 | 0.6225 |
| 500 (median 500) | (e) Neural | 8.6016 | n/a | 7.9637 | n/a |
| 2,000 (median 1,991) | (a) Rules scan | 1.3352 | 0.1062 | 0.1076 | 0.8220 |
| 2,000 (median 1,991) | (b) Local pipeline | 1.4251 | 0.1300 | 0.1293 | 0.9474 |
| 2,000 (median 1,991) | (c) Feature pipeline | 1.6099 | 0.2833 | 0.1950 | 2.5749 |
| 2,000 (median 1,991) | (e) Neural | 47.88 | n/a | 38.91 | n/a |
| 10,000 (median 9,984) | (a) Rules scan | 6.9908 | 0.5525 | 0.5510 | 3.9362 |
| 10,000 (median 9,984) | (b) Local pipeline | 7.3535 | 0.7289 | 0.6411 | 4.5751 |
| 10,000 (median 9,984) | (c) Feature pipeline | 8.7968 | 1.6976 | 0.9787 | 12.22 |
| 10,000 (median 9,984) | (e) Neural | 240.7 | n/a | 223.9 | n/a |
| 50,000 (median 49,985) | (a) Rules scan | 43.48 | 5.0354 | 2.9231 | 20.25 |
| 50,000 (median 49,985) | (b) Local pipeline | 46.49 | 7.0773 | 3.5454 | 23.06 |
| 50,000 (median 49,985) | (c) Feature pipeline | 59.02 | 11.97 | 5.4419 | 61.67 |
| 50,000 (median 49,985) | (e) Neural | 1,385.9 | n/a | 1,390.3 | n/a |

Steady-state rate, ms per 1,000 characters at the 10,000 /
50,000 tiers (median ms ÷ median chars × 1,000):

| Stage | Python | TypeScript | Rust | Swift (kit) |
| --- | ---: | ---: | ---: | ---: |
| (a) Rules scan | 0.700 / 0.870 | 0.055 / 0.101 | 0.055 / 0.059 | 0.394 / 0.405 |
| (b) Local pipeline | 0.737 / 0.930 | 0.073 / 0.142 | 0.064 / 0.071 | 0.458 / 0.461 |
| (c) Feature pipeline | 0.881 / 1.181 | 0.170 / 0.239 | 0.098 / 0.109 | 1.224 / 1.234 |
| (e) Neural | 24.113 / 27.727 | n/a / n/a | 22.427 / 27.814 | n/a / n/a |

Reading the curve:

- **The local stages are linear in length from ~500 chars.**
  Per-1,000-char rates at 500, 2,000 and 10,000 chars sit within
  ~±10% of each other in nearly every cell; per-call fixed costs
  are small (≤ ~0.05 ms) and dominate only the ~60-char reference
  row. Two exceptions at 50,000 chars: Python's local rates rise
  ~25–35% and TypeScript's roughly double, while Rust's and
  Swift's stay flat — consistent with span-volume-driven work
  (merge/render over ~1,000 spans per document) costing more in
  the Python and TS pipelines, though that mechanism is inferred,
  not profiled.
- **Neural cost steps up at the windowing transition, then runs
  linear in chunk count.** At 500 chars a document fits in one
  partial 382-token window (17.2 ms/1,000 chars, Python); from
  2,000 chars every window runs at full length and the rate
  settles at ~24 ms/1,000 chars, drifting to ~27.7 at 50,000.
  The drift tracks the tokenize bucket: its share of `predict()`
  grows from ~9% (500–2,000 chars) to ~15% (Python) / ~19%
  (Rust) at 10,000, because the splitter re-encodes text at each
  recursion level. Inference remains the majority (~60–85% of
  predict across tiers and languages); post-processing stays
  ≤ ~6%. (Stage decomposition was collected at the 500, 2,000
  and 10,000 tiers, not at 50,000.)
- **Python vs Rust neural converges to a tie at length:**
  8.60 vs 7.96 ms at 500 chars, 47.9 vs 38.9 at 2,000, 240.7 vs
  223.9 at 10,000 — and 1,385.9 vs 1,390.3 ms at 50,000. At
  document scale the host language disappears; it is all ONNX
  Runtime, in both.
- **At the top tier the neural layer is the pipeline:** a
  50,000-char record bundle costs ~1.4 s of neural per document
  in either SDK language, against 3.5–62 ms of local pipeline —
  97–99.7% of the full-pipeline time.

Digest gate at length — and a parity finding. Local modes
(scan, rules, features) match across all four implementations
at every tier, scored and no-score. Neural predict span sets
also match everywhere measured: after the chunk-boundary fix
described below, the no-score digests agree at every tier in
both neural modes. One standing qualification, new at length:

1. **Scored digests diverge at every tier ≥ 2,000** in both
   neural modes (they match at 500 and on the fixture notes).
   Span scores are means of per-token confidences; with hundreds
   to ~1,100 spans per document, some span's mean lands within a
   rounding unit of the 4-decimal digest format and the two ONNX
   session configurations round it differently. Bisected at
   50,000 chars: 7 of 20 documents contain at least one such
   span; the one examined in detail (doc 9) had exactly one —
   a NAME_PATIENT span over the word "Edema", 0.7364 (Python)
   vs 0.7363 (Rust), identical span set. This is a formatting-
   level sensitivity of the scored digest, not a span
   difference — but it means scored byte-parity, exact on the
   fixture corpus, does not survive contact with documents.
2. **One genuine output divergence, tier 2,000, 1 of 100
   documents — FOUND AND FIXED (2026-10-29).** In doc 59, an
   ADDRESS span ("Apt. 36") that begins at the second chunk's
   leading token scored 0.9202 in Python and 0.9903 in Rust.
   Root cause, pinned down: Python's `predict` shifts *every*
   tokenizer offset by the chunk offset — including the (0, 0)
   offsets of the [CLS]/[SEP] specials, which become
   (coff, coff) — and its decode skips only offsets exactly
   equal to (0, 0), so for chunks after the first the
   [CLS]-position prediction (here B-ADDRESS at 0.7096) opens
   the span and its confidence enters the score mean. Rust's
   offset construction stored (0, 0) for those positions
   unshifted, so its decode always skipped them and scored the
   span over the three content tokens only. The merge keeps the
   higher-scored of two overlapping ADDRESS spans, so the
   winner flipped: Python redacted "77 Harbor Blvd, Apt. 36",
   Rust redacted only "Apt. 36" and left the street name and
   number in the clear. The fix aligns Rust to the Python
   reference: `predict_chunk` now builds offsets through a
   `global_offsets` helper that shifts every encoding offset,
   with no (0, 0) special case. Regression coverage: a Rust
   integration test pins doc 59's Python spans and scores
   (failed pre-fix at 0.9903, passes post-fix), and unit tests
   pin the offset semantics. Post-fix verification: the tier
   2,000 full-pipeline no-score digest now matches Python
   (it was the only no-score mismatch at any tier), and
   no-score digests match at every tier in both neural modes;
   the scored digests still diverge at tiers ≥ 2,000 per
   Finding 1, which is unaffected. The digest fields in
   `bench/length_results.json` were regenerated for the neural
   sections of the 2,000 / 10,000 / 50,000 tiers; the timing
   runs are the original sweep's (the fix is an O(1)-per-token
   offset change and does not move them).

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
  it and confirms the hash matches the file on disk. The headline
  tables use those two corpora only; the 500–50,000-char length
  curve has its own seeded corpora and results file (see
  "Document length"). Each runner
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
  "a few hundred ms", not a precise figure.- The fixture notes are short — every headline number in this document is scoped to micro-text (~60 chars). The length curve is measured separately in "Document length"; on the 500-note clinical holdout used in the Presidio comparison, Python's full neural pipeline measured 8.4 ms/note p50 — document length dominates. That section also records two parity qualifications found at length: scored neural digests diverge between Python and Rust at tiers ≥ 2,000 chars (rounding-level; span sets identical — still standing), and one genuine chunk-boundary output divergence bisected at the 2,000 tier, fixed in the Rust decoder on 2026-10-29 (see Finding 2 there) — cross-SDK byte-parity is exact on the fixture corpus and, at length, exact for span sets with scored digests excepted per Finding 1.
