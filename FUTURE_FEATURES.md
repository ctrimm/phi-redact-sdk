# Future features

Features deliberately **not** in the SDK today, recorded so the scope decisions
stay explicit. The 0.4.0 parity build (batch, LLM scrub/restore, anonymize
operators, allow/deny lists) was scoped to English-language text redaction;
everything below was considered and deferred on purpose. An item graduates
from this file by: GitHub issue → decision in `DECISION_LOG.md` → build.

---

## 1. HIPAA Safe Harbor transformation mode — detailed spec

**Status:** specified, not built. Tracked in GitHub issue
[#1](https://github.com/ctrimm/phi-redact-sdk/issues/1),
"Feature: HIPAA Safe Harbor transformation mode".

### Motivation

Two findings drive this:

1. **Phi-redact's worst measured weakness is over-redaction.** In the
   October 2026 head-to-head against Microsoft Presidio
   (`phi-redact/evals/presidio-vs-phiredact/`), phi's public-geography
   retention was **0.133** — tuned Presidio scored **0.833**. Phi only knows
   how to delete or placeholder a span, so public geography that carries no
   privacy risk gets destroyed along with the identifiers.
2. **Safe Harbor is the de-identification method clinical compliance teams
   actually ask for.** HIPAA's Safe Harbor method (45 CFR 164.514(b)(2)) is
   generalization-based for several identifier classes: ZIP codes truncate,
   dates reduce to year, extreme ages band. Generalization preserves
   analytic utility — year-level trends, 3-digit-ZIP geography, age bands —
   that placeholders and deletion destroy. A redaction SDK aimed at clinical
   text that cannot speak Safe Harbor is missing the vocabulary its users'
   compliance reviews are written in.

### What Safe Harbor requires

45 CFR 164.514(b)(2) has two conditions:

- **(b)(2)(i)** — remove all 18 enumerated identifier types, with the
  specific generalization rules for geography, dates, and ages.
- **(b)(2)(ii)** — the covered entity must not have actual knowledge that
  the remaining information could be used, alone or in combination, to
  identify an individual. This is an attestation about the data and its
  context. **Software cannot satisfy or verify it** (see Honesty
  constraints).

### Transformation table

Phi-redact labels are the SDK's current taxonomy (neural + rules layers).
Handling is one of: **generalize** (replace with a coarsened value),
**remove** (delete, or placeholder under the active operator), or **n/a**
(not a text phenomenon).

| # | Identifier (164.514(b)(2)(i)) | Required handling | Phi-redact labels | Mode behavior |
|---|---|---|---|---|
| A | Names | Remove | NAME | Remove / placeholder (as today) |
| B | Geographic subdivisions smaller than a state (street address, city, county, precinct, ZIP, geocodes) | Remove sub-state geography. ZIP → first 3 digits, **except** ZIP3 areas whose combined population is ≤ 20,000 (per current Census data) → `000`. State itself may remain | ADDRESS-type spans, ZIP, city spans | **Generalize ZIP → ZIP3 or `000`**; remove street/city/county spans; state retained |
| C | All elements of dates except year (birth, admission, discharge, death, …) | Keep year only | DATE | **Generalize → year** (`03/14/1952` → `1952`) |
| C′ | Ages over 89, and date elements indicative of such age | Aggregate into a single category: age 90 or older | AGE (and DOB-derived ages) | **Generalize → `90+`**; ages ≤ 89 unchanged |
| D | Phone numbers | Remove **entirely** — Safe Harbor allows no area-code retention (that is Limited Dataset, a different method) | PHONE | Remove |
| E | Fax numbers | Remove entirely | FAX (currently often labeled PHONE by the model) | Remove |
| F | Email addresses | Remove | EMAIL | Remove |
| G | Social Security numbers | Remove | SSN | Remove |
| H | Medical record numbers | Remove | MRN | Remove |
| I | Health plan beneficiary numbers | Remove | HEALTHPLAN | Remove |
| J | Account numbers | Remove | ACCOUNT, BANK_ACCOUNT (and CREDITCARD as an account number) | Remove |
| K | Certificate / license numbers | Remove | LICENSE, DRIVERS_LICENSE (PASSPORT numbers are handled under R) | Remove |
| L | Vehicle identifiers and serial numbers, incl. license plates | Remove | VIN | Remove |
| M | Device identifiers and serial numbers | Remove | DEVICE | Remove |
| N | Web URLs | Remove | URL | Remove |
| O | IP address numbers | Remove | IP spans (rules layer) | Remove |
| P | Biometric identifiers (finger/voice prints) | Remove | — | n/a for text; media concern, see image redaction below |
| Q | Full-face photographs and comparable images | Remove | — | n/a for text; see image redaction below |
| R | Any other unique identifying number, characteristic, or code | Remove (the re-identification-code exception at 164.514(c) is **not** implemented by this mode) | IDNUM, GOVERNMENT_ID, PASSPORT, ROUTING_NUMBER, IBAN, TAX_ID, CUSTOM deny-list spans | Remove |

Notes on the table:

- Only three classes generalize: **ZIP, dates, and ages > 89**. Everything
  else removes. Any "generalization" beyond the table is not Safe Harbor
  and must not be labeled as such.
- Labels not among the 18 types (e.g. EMPLOYER, HOSPITAL, PROFESSION) are
  not Safe Harbor identifiers. The mode does **not** un-redact them: they
  keep whatever the active operator does today. The mode only overrides
  handling for the covered labels above.
- ZIP3 population rule: the practical data source is the list of restricted
  3-digit ZIP prefixes HHS publishes (derived from Census Bureau data;
  the count of restricted prefixes is small and changes rarely). The SDK
  would bundle the prefix list as static data with its vintage recorded,
  not fetch population data at runtime.

### API sketch

Safe Harbor is a **policy layer over span detection, not a new detector**.
Detection (rules + neural merge, thresholds, allow/deny lists) runs exactly
as today; a per-label transform table then decides what replaces each span,
superseding the anonymize operator for covered labels.

Python:

```python
from phi_redact import Redactor

r = Redactor(mode="safe_harbor")                 # year-only dates, ZIP3, 90+
result = r.redact(note)

# Optional, explicitly NOT Safe Harbor (see below):
r2 = Redactor(mode="safe_harbor", date_shift=True, salt="site-secret")
```

TypeScript: `new Redactor({ mode: "safe_harbor" })`, same options.

Composition rules:

- Allow/deny lists still apply first: allowlisted spans survive untouched
  (and the caller owns the compliance consequence); denylist spans are
  removed under row R.
- The transform table is pure data + pure functions, unit-testable without
  the model.
- Result metadata records that Safe Harbor mode was applied, the ZIP3 data
  vintage, and whether date shifting was on — so output is self-describing
  in an audit trail.

**Date shifting (alternative, separate transform).** Instead of reducing
dates to year, shift every date in a patient's records by one consistent
offset, derived deterministically from the hash salt + a patient key, so
intervals and record coherence survive across documents. This is a common
de-identification technique, but it is **not Safe Harbor** — shifted full
dates belong to the Limited Dataset / expert-certification world. It ships,
if at all, as an opt-in flag that is off by default, is documented as
non-Safe-Harbor wherever it appears, and disables the "safe_harbor" label
on result metadata when combined with full-date retention.

### Honesty constraints

- The SDK must **never** claim certified de-identification, and the mode
  must never be described as "HIPAA compliant" in README, docs, or
  docstrings. Wording is: "implements the mechanical transformations of
  45 CFR 164.514(b)(2)(i)".
- Condition (b)(2)(ii) (no actual knowledge of residual identifiability) is
  the user's attestation about their data and context. The docs must state
  the mode cannot and does not address it.
- The mode inherits the detector's recall. A missed span is an
  untransformed identifier: measured clinical full-value recall for v9 is
  0.969–0.980, which is not 1.0. Docs must say output is not guaranteed
  de-identified and remains subject to the user's review obligations.

### Evaluation plan

- Reuse the Presidio-comparison harness and its public-geography retention
  metric. Targets: retention from **0.133 toward ≥ 0.833** (tuned
  Presidio's score) on the same public set, with **no drop** in clinical
  full-value recall on the clinical holdout and golden-164.
- New Safe-Harbor-specific fixtures (deterministic, unit-level):
  restricted ZIP3 → `000`; populous ZIP3 → 3 digits retained; 5-digit ZIP
  never survives whole; dates → year only across formats (numeric, written,
  DOB/admission/discharge); age 94 → `90+`, age 89 unchanged; phone and fax
  removed in full (assert no area code survives); state name retained; city
  and street removed.
- Cross-SDK parity: identical transformed output Python vs TypeScript on
  the shared fixture set, as with the 0.4.0 operators.

### Open questions (for Cory)

1. **ZIP3 data source and vintage.** Bundle HHS's published restricted-ZIP3
   list as static data (simple, auditable, lags Census updates) vs compute
   from the Census ZCTA population file at build time (fresher, heavier
   pipeline)? And how often do we refresh it?
2. **Date-shift default.** Recommendation: off, always — Safe Harbor means
   year-only; shifting is a separate opt-in. Confirm?
3. **City names.** The regulation requires removing geographic subdivisions
   smaller than a state, so Safe Harbor mode removes city. Is a separate
   non-Safe-Harbor "generalize" variant that keeps city worth having, or
   strictly regulation-shaped only?
4. Edge case: state names inside facility names ("Texas Children's
   Hospital") — HOSPITAL spans are removed/placeholdered today anyway, but
   confirm that behavior is the intended interaction.

---

## Other deferred features

**Image redaction (OCR, incl. DICOM).** Presidio's lane today. Needs an OCR
engine decision and breaks the 11.3 MB footprint story that makes phi
deployable in-browser/on-device. Revisit only with a concrete user need.

**Structured data beyond JSONL.** JSONL batch processing shipped in 0.4.0.
CSV column redaction and FHIR resource traversal (redact by path, preserve
resource structure) are the natural next formats for clinical users; both
are SDK-layer work on top of the existing text engine.

**Input-side streaming redaction.** 0.4.0 shipped streaming-safe *restore*
(ChunkedRestorer) for LLM output. Streaming *input* — redacting text that
arrives in chunks, where a span can straddle a chunk boundary — needs an
overlap-window design and is unsolved.

**Compliance audit hook.** An optional callback/event stream emitting label
counts, span counts, mode, and model version per call — never values — so
deployments can keep a de-identification audit trail without logging PHI.

**Checksum-verified model downloads.** Pin SHA-256 digests for the ONNX
model and tokenizer in the SDK and verify on download/cache, matching the
version-pinning discipline the SDK already applies to itself.

**Spanish-US language support.** A model-side project (training data, eval
sets, threshold recalibration), not SDK work — the SDK absorbs a new model
through its label map once one exists. Phase two of the model roadmap.
