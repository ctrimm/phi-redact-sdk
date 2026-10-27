# phi-redact SDKs

Client libraries for **phi-redact** — compact on-device PII/PHI redaction.
Deterministic, checksum-gated rules layer for structured identifiers (cards,
SSN, IBAN, phone, email, URL, IP, ZIP, dates, MRN), plus an optional ONNX
neural layer for names, addresses, and free-text PHI the rules cannot see.

- [`python/`](python/) — `phi-redact` 0.5.0 on PyPI (planned). Module-level
  `redact()` / `scan()` / `restore()` (rules-only, zero dependencies) plus a
  `Redactor` class that loads an ONNX model for the neural layer.
- [`typescript/`](typescript/) — `phi-redact` 0.5.0 on npm (planned). Pure-TS
  port of the rules layer (zero dependencies, synchronous, Node 18+ and
  browsers) plus a `RemoteRedactor` HTTP client for the neural layer.
- [`rust/`](rust/) — `phi-redact` 0.6.0 on crates.io (planned). Native port
  of the full local pipeline (rules, operators, lists, recognizers,
  thresholds/review, streaming restore) accepting injected neural spans,
  with real ONNX inference behind the `neural` cargo feature. Byte-for-byte
  parity with Python is enforced in `e2e/parity/`; measured performance
  vs the other SDKs is in [`rust/PERF.md`](rust/PERF.md).

Merge policy: rule spans always win overlaps; neural spans overlapping a
rule span are dropped; placeholders are numbered per label in document
order (`[PHONE_1]`, `[NAME_PATIENT_2]`).

## Install

```bash
# Python — rules-only, zero dependencies
pip install "git+https://github.com/ctrimm/phi-redact-sdk#subdirectory=python"
# with the ONNX neural layer
pip install "git+https://github.com/ctrimm/phi-redact-sdk#subdirectory=python#egg=phi-redact[onnx]"

# TypeScript
git clone https://github.com/ctrimm/phi-redact-sdk
cd phi-redact-sdk/typescript && npm install && npm run build

# Rust
cd phi-redact-sdk/rust && cargo build --release   # add --features neural for ONNX
```

## Model

The current default model is **v9** (2026-10-21), published at
https://huggingface.co/CDT5058/phi-redact — download
`phi-redact-v9-int8.onnx`, `tokenizer.json`, and `config.json` from the
repo root and pass them to `Redactor` (see `python/README.md`). On the
164-case golden set (rules + neural, the production configuration, at
the SDK's default threshold of 0.6) v9 scores 0.974 full-value recall
and 0.746 precision, with a clean-case FP rate of 0.200 — it improves
on v8 on both axes at once (v8: 0.958 recall, 0.695 precision, 0.440
clean-FP). v8 and v7 remain available in the same HF repo under
versioned filenames (`phi-redact-v8-int8.onnx`, `tags-v8.json`,
`config-v8.json`, `model-v8.safetensors`).

## Links

- [Logging integrations (CloudWatch & Datadog)](docs/logging-integrations.md) — scrub logs at emit time or in the pipeline
- Demo and writeup: https://corytrimm.com/phi-redact
- Model and eval harness: https://huggingface.co/CDT5058/phi-redact

## Disclaimer

This tool redacts likely PII/PHI spans; it is not a certified
de-identification method and makes no HIPAA Expert Determination or Safe
Harbor claim. Validate on your own data before relying on it.

## License

Apache-2.0. See [LICENSE](LICENSE).
