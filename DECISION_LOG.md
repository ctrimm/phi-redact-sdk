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
