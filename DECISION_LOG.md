# DECISION_LOG.md

Data-backed decisions for this repo. Revisit an entry when new evidence
contradicts it; never let an unevidenced call harden into lore.

## 2026-10-18 — Repo is public

**Decision:** publish `ctrimm/phi-redact-sdk` as a public repo.
**Evidence:** Cory's call — "public since the model is public on Hugging Face"
(HF repo `CDT5058/phi-redact` already publishes the model card and the
164-case eval harness publicly). The SDKs contain no private data: the rules
layer is deterministic patterns, and the Python SDK ships without model
weights (the ONNX model is a separate download).
**Date:** 2026-10-18.

## 2026-10-18 — Standalone repo with vendored rules layer

**Decision:** the repo is self-contained — `python/redact/` vendors the
deterministic rules layer instead of referencing `../../redact` in the
sibling phi-redact project checkout.
**Evidence:** the original `pyproject.toml` mapped `redact = "../../redact"`,
which only resolves inside the full project checkout; `pip install` from
this repo alone would fail. Vendoring keeps one install story
(`pip install ...#subdirectory=python`) at the cost of syncing `rules.py`
when the upstream project changes it.
**Date:** 2026-10-18.

## 2026-10-18 — Kept the `phi-redact` name

**Decision:** no PII in the brand name; "PII/PHI redaction" is the tagline.
**Evidence:** Cory's naming question (2026-09-16) — recommendation was to keep
the short brand and carry discoverability in the package descriptions and
keywords (`pii`, `phi`, `redaction`), since renaming is only cheap before the
first registry publish. Both SDK descriptions already read "PII/PHI
redaction".
**Date:** 2026-09-16 (recorded 2026-10-18).

## 2026-10-18 — Neural special token ids read from the tokenizer

**Decision:** `OnnxEngine` resolves start/end marker ids from the loaded
tokenizer (`[CLS]`/`[SEP]` when present, otherwise the legacy `<s>`/`</s>`
fallback 0/2) instead of hardcoding `BOS, EOS = 0, 2`.
**Evidence:** Cory's report — v7's vocab has `[PAD]`/`[unused1]` at 0/2
and `[CLS]`/`[SEP]` at 101/102; the HF Space feeds 101/102, so the Python
neural layer was scoring below the model. Verified: the smoke tokenizer
has no `[CLS]`/`[SEP]` (falls back to 0/2, behavior unchanged for it); a
synthetic WordPiece vocab with `[CLS]`/`[SEP]` resolves 101/102. Covered
by `python/tests/test_neural.py`.
**Date:** 2026-10-18.

## 2026-10-18 — Phone pattern consumes a leading "(" or "+"

**Decision:** the PHONE rule's leading `\b` became `(?<![\w])`, so
`(415) 555-0132` and `+1 415-555-0132` match whole instead of leaving `(`
or `+` behind (`([PHONE_1]`, `+[PHONE_1]`).
**Evidence:** Cory's report. The `\+?` in the old pattern was effectively
dead code — `+` never sits at a word boundary, so the country code was
unreachable. Regression-guarded (`abc415-555-0132` still not matched),
applied identically to the TypeScript port, covered by the parity e2e.
**Coordination:** Cory's Swift port intentionally mirrors the SDK, so it
needs the same one-line change: replace the leading `\b` in its phone
pattern with `(?<![\w])`.
**Date:** 2026-10-18.
