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

## 2026-10-18 — Phone pattern also consumes "(" glued to a word char

**Decision:** the PHONE rule's start became `(?:(?<=\w)\(|(?<![\w]))`,
so `x(312) 555-0167` redacts to `x[PHONE_1]` instead of `x([PHONE_1]`.
**Evidence:** Cory's follow-up report; same bug class as the 2026-10-18
`(`/`+` fix. Validated against a 12-case battery first (`x(abc)` still
matches nothing, `abc415-555-0132` still rejected, `x(abc) 415-555-0132`
still finds the later phone). Applied identically to the TypeScript port;
parity e2e covers it (now 9 notes). The parity harness now fails fast
with "typescript/dist is stale" instead of a misleading span diff when
the TS build is older than the sources.
**Coordination:** same one-line Swift-port delta, extended: replace the
pattern's leading `\b` with `(?:(?<=\w)\(|(?<![\w]))`.
**Date:** 2026-10-18.

## 2026-10-19 — New deterministic rules: EIN, ABA routing, passport, DL, gov ID, bank account

**Decision:** added six rule families to both SDKs (Python + TypeScript,
parity-tested): TAX_ID (`\d{2}-\d{7}`, EIN format), ROUTING_NUMBER
(`\d{9}` gated by the ABA checksum — only ~1/10 random 9-digit strings
pass, so plain SSNs are not mislabeled), BANK_ACCOUNT (context-anchored
on "account", digits in capture group 1 so the span excludes the anchor),
PASSPORT (`[A-Z]\d{8}`), DRIVERS_LICENSE (`[A-Z]\d{7}` and `[A-Z]{2}-L\d{6}`),
GOVERNMENT_ID (`[A-Z]{3}\d{10}` USCIS receipt and `A\d{9}` alien number).
**Evidence:** golden 164-case eval. Rules-only full-value recall 0.354 ->
0.500; rules+v7 0.786 -> 0.885. Per-type: TAX_ID 0.000 -> 1.000, PASSPORT
0.167 -> 1.000, BANK_ACCOUNT 0.750 -> 1.000, GOVERNMENT_ID 0.000 -> 0.667
(the two dashed formats remain for the neural net), ROUTING_NUMBER and
DRIVERS_LICENSE now deterministic at 1.000. Precision held (0.749).
**Date:** 2026-10-19.

## 2026-10-19 — v8 training direction (no ai4privacy, Colab, no LLM verifier)

**Decision:** v8 trains on Google Colab from an extended synthetic generator
(`gen_synth_v8.py`); ai4privacy is excluded (its license is `license:other`,
commercial use requires contacting licensing@ai4privacy.com — Cory said do
not use it); no LLM judge/verifier tier.
**Evidence:** the eval pinpoints the gap as training data, not model
capacity: NAME 0.327 and ADDRESS 0.306 fail on conversational introductions
("my name is X") and address components the clinical-notes-only v7 corpus
never showed the model. TAX_ID/GOVERNMENT_ID were absent from training
entirely — no teacher distills what isn't in the data. gen_synth_v8.py adds
GOVERNMENT_ID, conversational name templates (particles, lowercase),
address-component spans, and boosted weak-label volume. The v8 notebook
(`phi_redact_colab8.ipynb`) pins marker ids from the tokenizer and gates
on the golden set (target: mid-90s full-value recall).
**Date:** 2026-10-19.

## 2026-10-19 — Red-team results: 16/27, "acct" anchor fixed, 2 known precision tradeoffs

**Decision:** red-teamed the "1.000" rule types with 27 adversarial cases
(`evals/rampart-vs-llm/redteam.jsonl`). Cory was right to be skeptical —
the 1.000s were on 4-6 easy cases. Found 8 misses, fixed 1 (bank account
"acct"/"ACCT-"/"account number" anchors now match in both SDKs), left 7
for v8 training (dashless/space-separated SSNs, hyphenated names, PO-box
and abbreviated addresses, dashed gov IDs — all covered by gravitee or the
v8 generator). Two false positives documented as inherent tradeoffs, not
fixed: the EIN pattern matches order numbers ("Order 12-3456789") and the
DL pattern matches confirmation codes ("Confirmation A1234567") —
context-gating would cost more recall than it saves.
**Evidence:** redteam.jsonl scoring: 15/27 -> 16/27 after the anchor fix.
**Date:** 2026-10-19.

## 2026-10-19 — Gravitee PII dataset added to v8 (Apache 2.0)

**Decision:** `gravitee-io/pii-detection-dataset` (175k examples, 781k
char-level spans, 25 PII classes, Apache 2.0 — license verified on the
dataset card) joins the v8 training mix via `convert_gravitee.py`
(PERSON->NAME_PATIENT, LOCATION->ADDRESS, US_SSN->SSN, US_ITIN->TAX_ID,
US_PASSPORT->PASSPORT, US_DRIVER_LICENSE->LICENSE, US_BANK_NUMBER->ACCOUNT,
etc.; 25k-row sample). It covers every neural gap in one package: PERSON
in 90/100 sampled rows, LOCATION in 75/100. The gretel PII set was passed
over — Apache 2.0 but no character offsets (spans would need fuzzy
string-matching).
**Evidence:** converter verified on 200 rows (770 entities, 0 bad spans,
exact offsets asserted).
**Date:** 2026-10-19.

## 2026-10-20 — v8 golden gate: 0.955 measured, mid-90s target hit; first S1 run was scored against stale Drive rules

**Decision:** v8 (student distilled from the distilbert teacher, rules +
merge as in prod) is accepted as passing the golden gate: full-value
recall 191/200 = 0.955, term recall 153/200 = 0.765 on the 164-case
golden set. Up from rules+v7 at 0.885 / 0.651. Per-type full: DL 6/6,
PHONE 19/19, ROUTING 4/4, GOV_ID 4/6, NAME 46/49, ADDRESS 35/36,
SSN 27/28. Remaining 9 full misses: GOV_ID x2 (the dashed/segmented
formats 1EG4-TE5-MK73 and HB-2024-88173 — known red-team targets),
NAME x3, ADDRESS x1, SSN x1, IP x1, URL x1.
**Evidence:** Colab S1 re-run 2026-10-20 after the rules sync (below).
The first S1 run (0.890 / 0.705) was measured against a stale Drive
rules.py (last modified 2026-09-22 — predating the Oct 18 phone
leading-(/+ fix, the Oct 19 DL/GOV_ID/BANK_ACCOUNT patterns, and still
carrying the Fed-district prefix gate in its ABA validator, which all
4 golden routing numbers fail). Diagnosis verified by downloading the
Drive file and diffing, and by the re-run landing exactly on the
derived corrected bound (191 full / 153 term, every per-type delta
matching prediction).
**Lesson:** before any Colab eval run, sync the canonical rules.py to
Drive and verify the copy the notebook pulled (size/checksum). The
notebook setup cell copies .py files from Drive; a live session keeps
the stale local copy until that cell re-runs.
**Date:** 2026-10-20.
