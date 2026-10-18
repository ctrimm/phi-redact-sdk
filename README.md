# phi-redact SDKs

Client libraries for **phi-redact** — compact on-device PII/PHI redaction.
Deterministic, checksum-gated rules layer for structured identifiers (cards,
SSN, IBAN, phone, email, URL, IP, ZIP, dates, MRN), plus an optional ONNX
neural layer for names, addresses, and free-text PHI the rules cannot see.

- [`python/`](python/) — `phi-redact` 0.1.0 on PyPI (planned). Module-level
  `redact()` / `scan()` / `restore()` (rules-only, zero dependencies) plus a
  `Redactor` class that loads an ONNX model for the neural layer.
- [`typescript/`](typescript/) — `phi-redact` 0.1.0 on npm (planned). Pure-TS
  port of the rules layer (zero dependencies, synchronous, Node 18+ and
  browsers) plus a `RemoteRedactor` HTTP client for the neural layer.

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
```

## Links

- Demo and writeup: https://corytrimm.com/phi-redact
- Model and eval harness: https://huggingface.co/CDT5058/phi-redact

## Disclaimer

This tool redacts likely PII/PHI spans; it is not a certified
de-identification method and makes no HIPAA Expert Determination or Safe
Harbor claim. Validate on your own data before relying on it.

## License

Apache-2.0. See [LICENSE](LICENSE).
