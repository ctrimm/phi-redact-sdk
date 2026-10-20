# Spec: checksum validation for neural spans

Status: SPEC — not implemented. Written 2026-10-20. Decision needed from
Cory before implementation. Applies to SDK 0.2.x (Python + TypeScript).

## Problem

The production merge policy is "rules win overlaps; neural spans pass
through otherwise." The deterministic rules layer validates its own
matches (Luhn, SSN structure, ABA checksum, IBAN mod-97), but neural
spans are never validated. The neural model therefore ships detections
that are *provably not* the identifier type claimed.

Measured on golden-164 with the shipped v8 INT8 artifact (2026-10-20 FP
taxonomy, 210 predictions / 56 false positives at precision 0.695):

- SSN false positives: `000-12-345`, `666-12-3456`, `900-12-3456` —
  area numbers 000/666/900+ are never issued.
- CREDITCARD false positives: `4170755268738198` (Luhn-invalid),
  `4685366999`, `15`, `829` (wrong length / fragment).
- Together: 9 of 56 FPs (~16%) are invalid-identifier hallucinations,
  most of them on otherwise clean cases, where they directly drive the
  clean-case FP rate (0.440 for v8).

These are the cheapest false positives in the system to remove: no
model change, no threshold change, no recall risk for any identifier
that could actually exist.

## Design

Add a validation gate for neural spans inside the merge step:

- Python: `phi_redact/engine.py`, in or immediately before
  `merge_entities(rule_ents, neural_ents)`.
- TypeScript: `typescript/src/engine.ts`, in `mergeEntities`.

Rules-origin spans are untouched (they are already validated at match
time). Only spans with `source == "neural"` whose label has a checksum
or structural validator are gated:

| Label | Validator (already in the rules layer) | Notes |
|---|---|---|
| SSN | `redact/rules.py:ssn_valid` / `rules.ts:ssnValid` | See normalization below |
| CREDITCARD | `luhn_ok` / `luhnOk` | Extracts digits itself; enforces 13-19 digits |
| IBAN | `iban_valid` / TS equivalent | ISO 13616 mod-97 |
| ROUTING_NUMBER | `aba_routing_ok` / TS equivalent | 9-digit ABA checksum |

All other labels pass ungated. In particular PHONE is NOT gated (no
checksum exists; the fictional 555-01XX range is format-valid), and
BANK_ACCOUNT / PASSPORT / DRIVERS_LICENSE / TAX_ID have no checksum to
gate on (the known `Order 12-3456789` TAX_ID false positive is a
context problem; it stays with the training-side fixes).

### SSN normalization

`ssn_valid` currently fullmatches only the dashed form
`DDD-DD-DDDD`. Neural spans may be bare 9-digit strings. The gate
normalizes before validating:

1. Extract digits from the span text.
2. If exactly 9 digits: apply the area/group/serial rules to the
   (3, 2, 4) slices.
3. If the span is the dashed form: existing `ssn_valid` applies.
4. Anything else (fragments like `000-12-345`, 8 digits): drop.

Implementation may add a small `ssn_digits_valid(digits)` helper next
to `ssn_valid` in both SDKs rather than complicating the existing
function; the two implementations must agree on every fixture.

### Configuration

- Python: `Redactor(..., validate_neural_spans: bool = True)` and the
  module-level `redact(...)` equivalent; the MCP server exposes it as a
  policy knob alongside threshold.
- TypeScript: matching option on the redact/merge options object.
- Default ON (recommendation). Rationale: a span that fails its type's
  checksum is never a true positive for that type; opting out exists
  for debugging and for downstream consumers who post-filter anyway.

### Suppressed-span visibility

Dropped spans must not vanish silently. Recommendation: count them in
the result object as `suppressed_neural_spans: int` (additive field,
no breaking change), and include the spans themselves only when a
caller passes `return_suppressed=True`. Open for Cory's call.

## Why this cannot hurt recall on real identifiers

The validators are format checks, not issuance checks:

- Every real SSN satisfies the area/group/serial structure.
- Every real payment card satisfies Luhn.
- Every real IBAN satisfies mod-97; every real routing number
  satisfies the ABA checksum.

The gate removes only values that cannot exist. The one class of
"loss" is deliberately fake test data that happens to be invalid —
which is the point.

## Expected impact (estimate, single-sample)

Golden-164, v8 artifact: removing the 9 invalid-identifier FPs moves
precision 0.695 -> ~0.726 (146 hits / 201 predictions) with zero
recall change, and clears the clean cases whose only predictions were
invalid identifiers. This is an estimate recomputed from one 164-case
sample, not a guarantee; the acceptance test below measures it for
real before release.

## Test plan

1. Unit fixtures (both SDKs, same strings):
   - Drop: `000-12-345`, `666-12-3456`, `900-12-3456`, bare-digit
     variants of the same, `4170755268738198`, `4685366999`, `829`.
   - Keep: a Luhn-valid test card (e.g. `4111111111111111`), a
     structurally valid SSN (e.g. `123-45-6789` — format-valid,
     not a real person's number), a valid IBAN test vector, a valid
     ABA routing number (e.g. `021000021`).
   - Rules-origin spans identical to the drop list must survive
     (gate is neural-only) — in practice rules never emit them,
     which the test asserts by construction.
2. Golden-164 re-score with the gate on vs off, same artifact:
   acceptance = recall identical, precision up, zero gold spans among
   suppressed spans.
3. Cross-SDK parity: the e2e parity harness gains the fixture set;
   Python and TypeScript must suppress identically.
4. Red-team set: no case may change outcome except the invalid-ID
   decoy cases improving.

## Rollout

- Minor version bump, both SDKs (0.2.0 -> 0.3.0), changelog entry
  naming the behavior change and the opt-out flag.
- DECISION_LOG entry with the before/after golden numbers.
- Model card note: the gate is inference-side and applies to any
  model version (v7/v8/v9 alike); it is complementary to the v9
  training-side fixes (near-miss decoys teach the same lesson to the
  model; the gate enforces it deterministically).

## Alternatives considered

- Training only (v9 near-miss decoys): reduces but cannot eliminate
  invalid outputs; a generative failure mode deserves a deterministic
  backstop — the same reason the rules layer validates its own spans.
- Raising the neural threshold: blunt; the threshold sweep shows it
  trades recall away and still leaves high-confidence invalid spans.
- Dropping the labels entirely: throws away real detections.

## Open questions for Cory

1. Default ON as recommended, or default OFF for one release?
2. Suppressed-span visibility: count-only (recommended) vs full list?
3. Ship gate with SDK 0.3.0 independently of v9, or bundle the
   announcement with the v9 model release?
