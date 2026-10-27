# phi-redact — Rust SDK

Rust port of the phi-redact redaction SDK, at feature parity with the
Python and TypeScript SDKs' 0.5.0 core surface (crate version 0.6.0,
tracking the repo's current release). Deterministic, checksum-gated
rules layer for structured identifiers (cards, SSN, EIN, IBAN, ABA
routing, phone, email, URL, IP, ZIP, dates, MRN, passport, driver's
license, government IDs, context-anchored bank accounts), the full
merge pipeline, anonymize operators, allow/deny lists, custom
recognizers, per-label thresholds with needs-review mode, and
scrub/restore with streaming restore.

Like the TypeScript SDK, the local pipeline accepts **injected neural
spans** — produce them with whatever runtime you like and pass them to
`redact()`. Real ONNX inference (the v9 int8 model) is available
behind the cargo feature `neural` (see "Neural layer" below).

## Install

```toml
[dependencies]
phi-redact = { path = "rust" }        # from a checkout of this repo
# or, once published: phi-redact = "0.6"
```

## Quick start

```rust
use phi_redact::{redact, RedactOptions, Redactor};

let r = Redactor::new();
let result = r.redact("Pt Maria Santos, SSN 234-56-7890, ph 415-555-0132.")?;
println!("{}", result.redacted);       // "Pt Maria Santos, SSN [SSN_1], ph [PHONE_1]."
println!("{:?}", result.mapping);      // {"[SSN_1]": "234-56-7890", ...}
assert_eq!(phi_redact::restore(&result.redacted, &result.mapping),
           "Pt Maria Santos, SSN 234-56-7890, ph 415-555-0132.");

// One-shot free functions mirror the Python module API:
let result = redact("call 415-555-0132", &[], &RedactOptions::default())?;
```

## API overview

- `scan(text) -> Vec<Entity>` — rules-only entity scan.
- `redact(text, neural_spans, options) -> Result<RedactionResult>` —
  the local pipeline: rules > recognizers > neural merge, allowlist
  filter, denylist injection, score partition (per-label thresholds +
  review margin), then the configured operator is applied.
- `Redactor` — a configured redactor (`Redactor::new()`,
  `Redactor::with_options(RedactOptions)`); methods `redact`, `scan`,
  `scrub` (returns `(redacted, mapping)`), `redact_batch`,
  `restore` (associated function), `engine_name()`.
- `redact_batch(&[String], &options)` — per-document results,
  order-preserving.
- Operators (`RedactOptions::operator`): `Numbered` (default,
  reversible `[LABEL_n]` + mapping), `Label` (`<LABEL>`), `Mask`
  (structured IDs keep separators + last 4; other values become
  same-length asterisks), `Hash` (`[LABEL_xxxxxxxx]`, first 8 hex of
  SHA-256 over `salt + "\x00" + label + "\x00" + value` — byte-identical
  tokens across all three SDKs), `Remove`.
- `allowlist` / `denylist` (+ `deny_label`, default `CUSTOM`): exact
  values (case-insensitive) are never redacted / are always redacted
  with source `denylist`; deny wins over allow and over detections.
- `Recognizer`: regex patterns under a built-in or custom label, base
  `score` (default 0.85), context words + `context_boost` within
  `context_window` chars (default 40), `require_context`, capture
  `group` selection (default 0), and a validator — a named checksum
  (`ValidatorName::Luhn/Ssn/Iban/Aba`) or a closure via
  `Validator::Func`. One idiom difference from Python: an unknown
  validator *name* is unrepresentable — `ValidatorName::parse` returns
  `Option`.
- `thresholds` + `review_margin`: per-label score gates for scored
  spans (neural + recognizer). A span in `[threshold - margin,
  threshold)` is returned on `result.review` as a `ReviewCandidate`
  instead of being redacted or dropped. Scoreless spans (rules,
  denylist) are never gated.
- `ChunkedRestorer`: streaming restore — `feed(chunk)` emits what is
  safe, holds back a trailing partial placeholder, `flush()` ends the
  stream; unknown placeholders pass through untouched.
- `phi_redact::rules` — the standalone deterministic layer, including
  `rules::redact_text`, which preserves the Python quirks (its
  placeholder counters run back-to-front; the engine numbers in
  document order instead).

All public offsets are **character indices** (Python semantics), not
byte offsets.

## Examples

```bash
cargo run --release --example batch_jsonl -- in.jsonl out.jsonl [--field text] [--operator hash --salt S]
cargo run --release --example parity_runner -- features ../e2e/parity/notes.jsonl
```

`batch_jsonl` mirrors `python/examples/batch_jsonl.py`: records whose
field holds a string are redacted in place and gain a `counts` object;
other records pass through; a malformed JSON line is a hard error.
Two formatting differences from the Python processor: output is
compact JSON (Python's `json.dumps` defaults insert spaces) and
non-ASCII is emitted as UTF-8 rather than `\uXXXX` escapes — records
are semantically identical.

## Parity

The port is verified byte-for-byte against the Python reference:

- `cargo test` — 104 tests ported from the Python/TS suites
  (validators, golden rule spans, phone-boundary cases, merge
  precedence, the shared hash vectors, mask strings, allow/deny,
  recognizers incl. context/validators/precedence, threshold/review
  edges, scrub/restore, chunked-restorer split points, batch).
- `e2e/parity/test_parity_rust.py` — the cross-SDK harness: the same
  16 feature configs + 2 batch configs × 15 fixture notes as the
  TypeScript harness, requiring identical redacted text, entity spans
  (start/end/label/source/placeholder/score), mappings, counts, and
  review lists vs Python.

Porting notes (where Rust forced a different mechanism for the same
semantics):

- The phone pattern relies on lookbehind (`(?<=\w)\(` /
  `(?<![\w])`), which the `regex` crate does not support. It is
  re-implemented as a body pattern plus manual previous-character
  checks, with finditer-compatible advancement; the Python suite's
  pinned cases (paren glued to a word char is consumed, number glued
  to a word is rejected) are in `tests/rules.rs`.
- The denylist's `(?<!\w)term(?!\w)` is re-implemented the same way.
- Recognizer/denylist patterns are compiled through a process-global
  cache — the analogue of CPython's `re` cache, which the Python SDK
  relies on implicitly. `regex::Regex::new` costs ~0.2 ms per pattern
  (Unicode tables); uncached, it dominated every recognizer redaction
  (measured: features pipeline 0.50 ms/note before the cache,
  0.012 ms/note after — see `../bench/` and `PERF.md`).
- Python's `round(x, 4)` (banker's rounding) vs this crate's
  half-away rounding can differ on exact .5 ties in the fifth decimal
  of a recognizer score — a theoretical edge that does not occur in
  the fixture corpus.

## Neural layer (feature `neural`)

```rust
let r = Redactor::new().with_model(
    "phi-redact-v9-int8.onnx", "tokenizer.json",
    Some("tags.json"),           // or a config.json with id2label
    phi_redact::DEFAULT_MAX_LEN)?;
assert_eq!(r.engine_name(), "rules+onnx");
```

`phi_redact::neural::OnnxEngine` ports the Python engine: chunking at
`max_len` tokens with 96-char overlap and the degenerate-split guard,
the serialized tokenizer truncation cap stripped before load, marker
ids read from the vocabulary ([CLS]/[SEP], else 0/2), only the
graph's declared inputs fed, BIOES decode with whitespace trim,
mean-token scores, per-chunk gating, and overlap dedupe keeping the
best score. Labels come from a config.json (`id2label`/`label2id`) or
a tags.json-style BIOES array; the default is `config.json` beside
the model (or `$PHI_REDACT_LABELS`).

**Sandbox status (2026-10-27): verified working, via a detour.** The
`ort-sys` build script normally downloads the prebuilt ONNX Runtime
from `cdn.pyke.io`; that host returns **HTTP 403** through this
sandbox's egress proxy (deterministic across retries, and the
download happens even with the `load-dynamic` feature). The working
combination here was:

```bash
export ORT_SKIP_DOWNLOAD=1   # build-script escape hatch in ort-sys
export ORT_DYLIB_PATH=/path/to/libonnxruntime.so.1.30.0
cargo build --release --features neural
```

with the dylib taken from the Python venv's onnxruntime 1.30 wheel
(`ort` binds the 1.22 C API; the 1.30 runtime serves it). On an
unrestricted network the plain `cargo build --features neural`
should take the normal download path instead. Either way the result
is verified, not assumed: `tests/neural.rs` pins the Python engine's
spans for three fixture notes against the real v9 int8 model — same
spans, same texts, scores equal to the recorded Python values — and
the perf sweep (`PERF.md`) ran the full Rust neural pipeline at
1.63 ms/note median vs Python's 1.53 on the fixture notes.

## Not ported

- The Python **logging kit** (`phi_redact.integrations.logging_kit`,
  the Lambda log scrubber) is Python-only by design (stdlib logging /
  structlog / Lambda Python runtime) and is out of scope here, as it
  is for TypeScript.
- The TypeScript SDK's `RemoteRedactor` HTTP client has no Rust
  counterpart; callers can POST to the serve endpoint with any HTTP
  client and inject the returned spans into `redact()`.
- Model download/verification helpers do not exist in any SDK; all
  three load local files.
