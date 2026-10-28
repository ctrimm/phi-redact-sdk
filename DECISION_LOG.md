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

## 2026-10-20 — SDK default model v7 -> v8; SDKs bumped to 0.2.0

**Decision:** the SDKs' current/default model is now v8. Python SDK and
TypeScript SDK bumped 0.1.0 -> 0.2.0 (TS is functionally unchanged —
rules/remote only; the bump keeps the pair in lockstep and marks the
model-default change). READMEs now name the v8 files on Hugging Face
(`phi-redact-v8-int8.onnx` + `tokenizer.json` + `config.json` from
CDT5058/phi-redact) as the ones to wire into `Redactor`.
**Evidence:** v8 INT8 ONNX scored through the shipped SDK on the
164-case golden set: full-value recall 0.958 / term 0.760 / precision
0.695 / strict F1 0.617 vs v7 INT8 at 0.885 / 0.651 / 0.749 / 0.435
(same scorer). The PyTorch checkpoint's Colab gate score was 0.955, so
INT8 quantization costs no recall. Tradeoff recorded, not hidden:
clean-case FP rate 0.440 (v8) vs 0.120 (v7) — v8 was recall-tuned.
Threshold 0.8 would give clean-FP 0.320 at 0.932 full-value recall;
the default stays 0.5, the gate threshold.
**HF layout:** repo root swapped to v8 (config.json, tags.json,
model.safetensors, README card); v7 preserved as phi-redact-v7-int8.onnx
(untouched), model-v7.safetensors, config-v7.json, tags-v7.json. Hub
weights verified bit-exact against the local checkpoint (logit diff 0).
Note the label sets differ: v7 = 121 tags, v8 = 129 (adds
GOVERNMENT_ID and TAX_ID, inserted mid-list — indices after FAX shift,
so v7 consumers must not pair the v7 ONNX with the root v8 tags/config;
the versioned v7 files exist for exactly that).
**Export note:** the Colab T1 export failed (optimum not installed);
the shipped INT8 was produced locally instead (torch.onnx classic
exporter + onnxruntime dynamic quantization), validated against PyTorch
(max logit diff 1.34e-05) and by the golden eval above.
**Date:** 2026-10-20.

## 2026-10-20 (later) — v8 artifact parity with v7 + demos updated
- Decision: give v8 the full v7 artifact set on HF and move both demos to v8.
- Evidence: HF now carries phi-redact-v8.onnx (fp32), phi-redact-v8-fp32.mlpackage
  (fp16, spec v9, weight.bin 22,149,378 B vs v7 22,145,266), phi-redact-v8.mlpackage
  (4-bit kmeans palettization, per-grouped-channel group_size=2, weight.bin 6,347,778 B
  vs v7 6,250,098), phi-redact-v8-fp32.tflite (44,482,960 B vs v7 44,472,464).
  Contracts were read from the v7 files first (Core ML: int32 inputs [1,256], float16
  logits; TFLite: serving_default int64 inputs [1,256], float32 out) and matched.
- Verification split: TFLite was executed in the LiteRT interpreter on Linux —
  max logit diff vs the FP32 ONNX 1.43e-05, argmax agreement 1.000. Core ML
  packages are spec-validated only (inputs/outputs/dtypes/shapes/spec version);
  Core ML cannot execute off macOS, so first run on a Mac remains the real gate.
- Conversion notes: coremltools 9.0 torch.export path needs transformers 4.57.x
  (5.x masking emits new_ones the EXIR frontend rejects); litert-torch needs
  enable_x64=True or inputs silently become int32; ai_edge_tensorflow's 268 MB
  wheel only downloads reliably via curl -C - resume (pip died at 228 MB 6x).
- Demos: HF Space CDT5058/phi-redact-demo now defaults to the v8 ONNX (it had
  been pairing the v7 model with the v8 root tags.json — mislabeling). Website:
  PR ctrimm/personal-site-astro#97 swaps the self-hosted model files + page copy
  (32 types, v7 tables labeled v7, v8 golden note). Cory merges/deploys.

## 2026-10-20 (later) — v8 run on Nemotron-PII: micro F1 0.849 vs v7 0.860
- Method recovery, not reinvention: the eval set is the exact file the v7 run
  used — Drive nemotron-eval.jsonl (convert_nemotron.py, test split, first
  2,000 docs, labels pre-mapped): 2,000 docs / 13,134 gold spans, matching
  the card fingerprint exactly. All docs fit the 256-token window.
- Harness validated before trusting it: train/evaluate.py score_model
  (argmax, BIOES decode, strict char-span) run on the preserved v7 weights
  reproduced every published v7 per-entity F1 to the third decimal and
  micro F1 0.860. Only then was v8 run through the identical code path.
- Result (model-only): v8 micro P 0.867 / R 0.832 / F1 0.849 vs v7 0.877 /
  0.843 / 0.860. A wash on this synthetic distribution (v7's training mix
  included a large Nemotron train conversion). v8 gains concentrate on
  the clinical-note distribution (golden 0.885 -> 0.958). Per entity: SSN
  0.933 vs 0.908, AGE/DEVICE/HEALTHPLAN up; EMPLOYER/LICENSE/CITY/DATE down.
- Filed: HF model card v8 section; website PR #97 amended (page note + PR
  body); raw results at phi-redact work/nemotron/results_{v7,v8}.json.

## 2026-10-20 — v9 training plan: buy precision back with negative training (data-only changes)
- Decision: v9 = v8 recipe unchanged (distilbert teacher 10 ep, bert-mini student 60 ep, alpha 0.7), data changes only, so deltas are attributable. Notebook phi_redact_colab9.ipynb (Drive, 25 cells); Cory runs it.
- Evidence for the changes (v8 measured 2026-10-20): golden precision 0.749 -> 0.695 and clean-case FP 0.120 -> 0.440 vs v7, FPs confident (threshold 0.9 barely helps); Nemotron-PII F1 0.860 -> 0.849; two segmented GOV_ID golden cases still missed.
- Changes: (1) hard negatives mined with checkpoint-student-v8 over gold-empty, rules-filtered training docs (mine_negatives_v9.py; caps 10k firing + 10k silent); (2) gen_synth_v9.py: segmented/dashed GOV_ID ~55%+ of GOV_ID samples (fresh values, shapes only — golden stays sealed) + 15% sparse low-density notes; (3) Gravitee cap 25k -> 15k; (4) Nemotron becomes a scored gate (S3).
- Pre-registered gates: golden recall >= 0.950 AND precision >= 0.720 AND clean-FP <= 0.250; Nemotron micro F1 >= 0.849 (stretch 0.860); red-team >= v8 baseline (S0 measures it; none existed). Export via export_onnx_local.py (no optimum — v8's T1 failure); benchmarks via eval_checkpoint.py (AutoModel — evaluate.py CLI still hardcodes Roberta).
- Catch during prep: multiconer-train.jsonl was listed in v8's Q3 sources but never existed on Drive — v8 silently skipped it. v9 drops it from all lists and the notebook's setup cell asserts every required input exists (plus canonical rules.py md5) before training.

## 2026-10-20 (addendum) — v8 FP taxonomy; v9 notebook patched; neural-span validation spec'd
- FP taxonomy of the shipped v8 INT8 on golden-164 (210 preds, 56 FPs, reconciles with precision 0.695): 46% CITY/STATE/ZIP over gold KEEP spans (public retention 0.133 vs v7 0.267 — v8's worst regression axis); 30% invalid identifiers on clean cases (never-issued SSN areas 000/666/9xx, Luhn-invalid cards, 555 numbers, bare street numbers); 20% NAME/ADDRESS boundary errors; prose person/place FPs a minority.
- Consequence for v9 (notebook patched before Cory's run, Drive copies md5-verified): gen_synth_v9 gains public_geo notes (geo rendered unlabeled in public/facility contexts — context discrimination, not geo suppression; product keeps CITY/STATE/ZIP as toggleable categories), prose_address notes (street labeled, accompanying geo unlabeled), and an expanded near-miss decoy pool (invalid SSNs/cards/555/bare numbers, ~35% of notes on top of v8's own). Notebook gains S4: ship-threshold sweep (0.5-0.8 via evaluate_golden --threshold), chosen by the gates, not by default.
- Spec'd, not built: docs/neural-span-validation.md — gate neural SSN/CREDITCARD/IBAN/ROUTING_NUMBER spans on the rules layer's existing checksum validators at merge time (merge_entities / mergeEntities). Measured upside on golden-164: 9 of 56 FPs removed, precision ~0.695 -> ~0.726, zero possible recall loss on identifiers that can exist. Awaiting Cory's answers to the spec's three open questions (default on/off, suppressed-span visibility, ship timing).

## 2026-10-20 — v10 plan written (V10_PLAN.md in the phi-redact project + Drive)
- Scope: structural levers beyond the v8/v9 recipe. WS1 teacher-ensemble silver corpus (fastino GLiNER2-PII + openai/privacy-filter + gliner_large arbiter + local Qwen3 judge; pseudo-labeling, all Apache-2.0; API outputs barred as training signal). WS2 real-world measurement (i2b2 eval-only under DUA; harness ~1 day post-access). WS3 gated ablations: CRF head (Viterbi in the SDK decoder, ONNX graph unchanged), label-smoothing off, capacity probe under the 23M-param bar, max_len 384. WS4 data diversity (LLM synthetic with verbatim span verification — spend/ToS decision; mix label-noise audit). WS5 inference composition (validator implementation, per-label thresholds, hosted ensemble tier spec). WS6 release parity per the v8 checklist.
- Draft v10 gates include two new headline repairs: public retention >= 0.500 (v8: 0.133) and Nemotron >= 0.860 (v8: 0.849), plus golden recall >= 0.960 / precision >= 0.750 / clean-FP <= 0.200 / strict F1 >= 0.680 at <= 12 MB INT8.
- Pending Cory decisions D1-D7 (DUA, LLM-generation budget, capacity probe, CRF, validator defaults, hosted tier, teacher discipline) — recommendations recorded in the plan.

## 2026-10-20 (addendum 2) — i2b2 declined; v9 pre-run shore-ups
- D1 decided (Cory): i2b2/N2C2 NOT pursued — DUA legwork not worth it for an eval-only benchmark. V10 WS2 rescoped in V10_PLAN.md to an independently generated, sealed synthetic eval set (breaks same-generator bias) + existing public benchmarks.
- v9 shore-ups applied before Cory's overnight run, all aimed at the worst FP-t taxonomy buckets: public_geo share 7% -> 10% and prose_address 6% -> 8% (geo-over-KEEP was 46% of FPs; retention 0.133), v9 decoy append rate 35% -> 50% (invalid identifiers were 30% of FPs), negative-mining caps 10k/10k -> 15k/15k. Training knobs deliberately untouched (label smoothing etc. stay WS3 ablations — one untested knob in a single overnight run makes regressions uninterpretable). Notebook rebuilt (27 cells) and Drive copies md5-verified.

## 2026-10-21 — v9 verdict: ALL pre-registered gates PASS; ship recommended at threshold 0.6
- Golden (INT8, SDK scorer): full-value recall 0.974 @th0.5 AND @th0.6 (torch S1 0.975; gate >=0.950 PASS; v8 0.955), precision 0.733 @0.5 / 0.746 @0.6 (gate >=0.720 PASS; v8 0.695), clean-FP 0.200 (gate <=0.250 PASS; v8 0.440), term recall 0.828 (v8 0.765), strict F1 0.677 @0.6 (v8 0.634). Nemotron-PII micro F1 0.849 (gate >=0.849 PASS at the line; P 0.870/R 0.830; stretch 0.860 missed). Red-team 0.870 full vs v8 0.826 PASS. S2: no material regression (Learning Agency 0.711->0.729; FewNERD/MultiNERD flat; WikiANN -0.011; govsynth 1.000).
- The v8 recall-precision tradeoff is broken: v9 improves recall AND precision simultaneously. Threshold call: 0.6 dominates 0.5 on the artifact (precision +0.013, strict F1 +0.007, retention 0.033->0.133, recall/clean-FP unchanged) — ship config = th 0.6.
- Honest negative: the public-geo lever did NOT move public retention (0.133, same as v8; keep-geo FPs 35->29 count, now 50% of a smaller FP pool). The generator's public contexts were too stylistically distinct from golden's keep contexts; the model learned the template, not the rule. Remaining FP mass: keep-context over-redaction (geo + DOB-adjacent). Invalid-ID and boundary buckets shrank as designed (pure 23->15, partial 15->6). Next lever for retention is v10 WS1 (silver corpus contextual variety), not more generator templates.
- eval_checkpoint.py CUDA device bug found by the Colab run (model never moved to device), fixed + re-uploaded; Colab re-run confirmed all numbers (Nemotron identical to local provisional: 0.849). Process note (Cory): run evals on the Colab box, not local CPU.
- Ship execution (HF push, SDK threshold default 0.6 + version bump, artifact parity, card) awaits Cory's word, as with v8.

## 2026-10-21 — v9 SHIPPED (Cory's word)
- HF repo CDT5058/phi-redact: root is v9 (model.safetensors/config/tags replaced; sha256-verified against local). v8 preserved as model-v8.safetensors, config-v8.json, tags-v8.json, alongside its already-versioned artifacts. Added phi-redact-v9.onnx, phi-redact-v9-int8.onnx, phi-redact-v9-fp32.tflite (argmax agreement 1.000 vs fp32 ONNX), phi-redact-v9-fp32.mlpackage (22.2 MB) and phi-redact-v9.mlpackage (5.6 MB; spec contract matches v7/v8). Model card rewritten for v9 (all gates, threshold rationale, retention negative result disclosed).
- SDK 0.3.0: DEFAULT_THRESHOLD 0.5 -> 0.6 (python engine; pinning test updated), READMEs name v9. Tests: python 60 passed, typescript 35 passed.
- HF Space (phi-redact-demo) redact-core.js default modelFile -> phi-redact-v9-int8.onnx; verified live.
- Conversion env note: Core ML export requires transformers 4.57.x (transformers 5 masking_utils emits aten new_ones, unsupported by coremltools 9.0's TorchExport frontend) + torch 2.9; TFLite/litert wants current torch + torchao. Isolated venvs at ~/workspace/tooling/{convert,tflite}-venv.

## 2026-10-25 — SDK 0.3.1: tokenizer truncation silently capped the neural scan (bug fix)

- **Bug** (found 2026-10-24 while building the Chrome extension, which ports this engine): the published tokenizer.json carries a training-side truncation setting (max_length 256). `OnnxEngine` loaded the tokenizer as-is, so `encode()` returned at most 256 ids. `_split`'s length check read that truncated length, chunking (max_len 384) never engaged, and `predict` re-encoded through the same cap — text past roughly token 254 was never scanned by the neural layer. The rules layer still covered the tail, so redacted output looked plausible; only neural-only entities (names, hospitals, addresses in prose) past the cap were silently missed.
- **Evidence** (real v9 int8 model, synthetic 670-token note with three physician names at ~10 / ~310 / ~610 tokens): pre-fix, the engine tokenizer reported 256 tokens for the document and the two names past the cap produced no entities; post-fix, all three names are detected, including spans at chars 3591-3613 of 3,642. Short-text parity: the standard 12-detection clinical note produces byte-identical output before and after the fix (git-stash A/B).
- **Fix:** `_load_tokenizer()` strips the `truncation` block from the serialized tokenizer config before loading. Chosen over the tokenizers library's runtime toggles because their shape changed across releases (attribute assignment pre-0.22; `no_truncation()` gone by 0.23) — config editing works on all supported versions (>=0.15). The engine's own chunking is now the only length limit, as designed.
- **Adjacent latent bug, fixed in the same change:** `_split` could recurse forever when the 96-char overlap window covered the whole left piece (back == 0 re-splits identical text). Unreachable at the default max_len — an over-budget chunk is far longer than 96 chars — but reachable with a small custom max_len. The split now drops the overlap in that degenerate case; found by the new regression test's toy geometry.
- **Tests:** 3 regression tests added (loader strips truncation; long text fully scanned at correct offsets with a scripted fake session; short text unchanged). Python 63 passed (was 60), TypeScript 35 passed (no TS code change — the TS local engine is rules-only and its neural path runs through this Python engine via the server). The Chrome extension's deliberate divergence (truncation off, chunked at 384) now converges with the SDK again.
- **Version:** 0.3.0 -> 0.3.1 (pyproject, typescript/package.json in lockstep). Also fixed version drift: `phi_redact/__init__.py` still said 0.2.0; now 0.3.1.

## 2026-10-27 — SDK 0.4.0: Presidio-parity features (English text only)

- **Decision:** close the SDK-layer gaps named in the 2026-10-26 Presidio comparison (evals/presidio-vs-phiredact/PRESIDIO_USE_CASES.md) in both SDKs, with identical semantics: (1) batch pipelines — `Redactor.redact_batch` / `redact_iter` (Python), `redactBatch` (TS), plus dependency-free JSONL processors (`python/examples/batch_jsonl.py`, `typescript/examples/batch_jsonl.mjs`); (2) LLM scrub-and-restore as a first-class pair — `Redactor.scrub()` (neural pipeline included) and a `ChunkedRestorer` in both SDKs that buffers trailing partial placeholders, restores split placeholders whole, and passes unknown/partial ones through untouched; (3) anonymize operators — `numbered` (default, reversible), `label` (Presidio-style `<LABEL>`), `mask`, `hash`, `remove`; (4) allow/deny lists on the Redactor / redact options.
- **Scope line (explicit):** English-language text only. Image redaction (OCR/DICOM), structured-data anonymization, and non-English / international recognizers are OUT of scope — Presidio keeps those lanes, and the SDK READMEs say so. SDK-layer work only: no model changes, no threshold changes.
- **Operator semantics, chosen once and mirrored:** only `numbered` returns a populated mapping; the other operators return an empty mapping (irreversible by design, documented in both READMEs). Mask: structured-ID labels (SSN/card/phone/kin) with value length >= 8 keep the last 4 chars, preceding alphanumerics become `*`, separators preserved; every other value masks to its own length. Hash: token `[LABEL_<digest>]`, digest = first 8 hex of SHA-256 of `salt \0 label \0 value` (UTF-8), salt a Redactor option defaulting to `""`; READMEs warn that unsalted hashes of low-entropy values are dictionary-attackable. TS implements SHA-256 in pure TS (zero-dependency, browser-safe; no node:crypto in src). Five shared vectors are pinned in both suites, and the cross-SDK parity harness runs the hash operator end-to-end.
- **Allow/deny semantics:** allowlist = case-insensitive exact match on span text, span dropped. Denylist = case-insensitive, word-boundary-aware exact matches injected as spans with source `denylist` and label = `deny_label` option, default `CUSTOM` (chosen over reusing an existing taxonomy label so deny hits stay auditable and never masquerade as model detections). A deny span wins any overlap (overlapping detected spans are dropped); deny wins over allow; processing order is allowlist filter, then denylist injection, after the existing rules-win merge.
- **Evidence:** Python 99 passed (63 pre-existing + 36 new), TypeScript 69 passed (35 + 34 new), cross-SDK parity e2e 2 passed — the new feature-parity test compares redacted text, entity spans (incl. source/placeholder), mappings, and counts across 10 feature configs x 11 fixture notes and requires byte-identical output from both SDKs. Note: the pre-existing Python baseline actually had 1 failure, not 0 — `test_version` pinned `__version__ == "0.2.0"` while the package was already 0.3.1; the 0.4.0 bump resolves the pin.
- **Status:** built and tested locally only. NOT pushed — the push is Cory's call, per standing rule.

## 2026-10-27 — SDK 0.5.0: custom recognizers + per-label thresholds with needs-review mode

- **Decision:** port the two remaining Presidio SDK-surface features Cory picked from the post-0.4.0 shortlist, in both SDKs with identical semantics: (1) a custom recognizer API — Python `PatternRecognizer` dataclass, TS `Recognizer` options object — regex patterns under built-in or custom labels, base score (default 0.85), Presidio-style context enhancement (context words within a per-side character window, default 40, raise the score by `context_boost`, default 0.10, capped at 1.0; `require_context` drops uncontextualized matches), and an optional validator (a user callable/predicate, or a named built-in checksum: `"luhn"`, `"ssn"`, `"iban"`, `"aba"` — reusing the rules layer's validators); (2) per-label thresholds (`thresholds={label: gate}`) plus a needs-review mode (`review_margin`): a scored span below its effective threshold but at/above `threshold - review_margin` is not redacted and not dropped — it is returned on `RedactionResult.review` / `result.review` ({start, end, label, text, score, source}) for human adjudication, the surface Presidio lacks.
- **Precedence (chosen, documented in both READMEs):** merge order is rules > recognizers > neural. Recognizers are deterministic user intent, so they win overlaps against neural spans like rules do; against rule spans they yield — the built-in rules are treated as "registered first". Among recognizers, first-registered wins (then pattern order, then document order), resolved inside `recognizer_spans` before the merge. Recognizer spans respect the allowlist (protected values drop) and lose overlaps to denylist injections (deny still wins), and they redact under the configured anonymize operator like any other span. Span source is `"recognizer"` in both SDKs.
- **Threshold semantics (chosen):** thresholds gate ONLY spans carrying a score (neural + recognizer), applied uniformly by final score. Spans without a score — the checksum-gated rules layer and denylist injections — are never gated and never become review candidates; a per-label threshold on `"SSN"` deliberately does not stop the SSN rule. This was the "weird edge" in the brief, resolved in favor of keeping the deterministic layer deterministic. The Python engine now asks the ONNX layer for spans down to a computed floor (min of global/per-label thresholds, minus the review margin) via `OnnxEngine.predict(text, threshold=...)`, so the partition — not the detector gate — decides redact vs review vs drop; global default threshold unchanged (0.6). Review candidates are excluded from the mapping and counts, ride along per-result in batch mode, and are ignored by restore (they were never replaced). `RedactionResult.to_dict()` gains a `"review"` key (additive; the serve contract test uses a superset check).
- **Safe Harbor mode:** deferred at Cory's direction — spec'd as a tracked future feature in FUTURE_FEATURES.md with a GitHub issue, not built in 0.5.0.
- **Evidence:** Python 140 passed (99 baseline + 41 new in `tests/test_recognizers_thresholds.py`, incl. a scripted OnnxEngine subclass proving the floor is handed to the neural layer and recognizer-vs-neural precedence end-to-end), TypeScript 109 passed (69 + 40 new in `test/recognizers.test.js`), cross-SDK parity e2e 2 passed — the harness grew from 10 to 17 feature configs x 15 fixture notes and now compares redacted text, entity spans (incl. source, placeholder, AND score), mappings, counts, and review lists byte-identically across both SDKs. Version bumped 0.4.0 -> 0.5.0 everywhere (pyproject, `__init__`, package.json, package-lock, TS VERSION, version pins, `to_dict` key-set pin).
- **Status:** built and tested locally only. NOT pushed, NOT committed — the push is Cory's call, per standing rule.

## 2026-10-27 — SDK 0.6.0: logging kit (Python only)

- **Decision:** make phi-redact work with backend logging systems (CloudWatch, Datadog) as a Python-only SDK addition: `phi_redact/integrations/logging_kit.py` with `RedactingFilter` (stdlib `logging.Filter`), `add_redacting_filter`, and `make_structlog_processor` (duck-typed; structlog is never imported), plus `python/examples/lambda_log_scrubber.py` — a CloudWatch Logs subscription-filter Lambda that scrubs each event batch and forwards to stdout (default; the Lambda's own log group becomes the scrubbed store), Datadog's HTTP intake (stdlib urllib), or a clean log group via lazily-imported boto3 (present in the Lambda runtime; deliberately not an SDK dependency). The Redactor is configured from env (`PHI_REDACT_MODEL`/`TOKENIZER`/`OPERATOR`/`SALT`/`THRESHOLD`), rules-only when no model is set. Packaging fix included: `phi_redact.integrations` added to pyproject's explicit package list, or the subpackage would have been missing from built distributions.
- **Fail-open filter (chosen, documented in the module docstring):** if redaction raises, the original record passes through unchanged. Rationale: a redaction bug must never take down the application's logging — or the application. The tradeoff is stated plainly in the docstring and README: a silent scrubbing failure can leak exactly what the kit exists to remove, so coverage should be monitored outside the log path, not by failing log calls. The structlog processor and the Lambda's per-event handling follow the same policy (the Lambda counts failures in its return dict rather than dropping the batch).
- **Placement semantics (chosen framing, docs + guide):** emit-side scrubbing (the filter, inside the producing process) is the only placement where PII never reaches storage at all. Pipeline-side scrubbing (the Lambda, batch passes) protects everything downstream of the pipeline, but CloudWatch stores first and filters on delivery — raw lines remain in the source log group. The docs say this rather than letting "works with CloudWatch" imply otherwise.
- **Python-only (chosen):** the logging-filter ecosystem that asked for this is Python (stdlib logging, structlog, Lambda Python runtime). The TypeScript SDK has no equivalent host ecosystem in scope; TS services scrub in their logging transport with the existing `scrub()` API. The integration guide (`docs/logging-integrations.md`) covers that pattern; a TS kit is future work if demand appears. TS package version unchanged (stays 0.5.0); Python bumps 0.5.0 -> 0.6.0.
- **Evidence:** Python 158 passed (140 baseline + 18 new in `tests/test_logging_kit.py`: message/args/literal scrubbing, handler- and logger-level attach, exception-traceback scrubbing incl. the scrub_exc=False counter-case, fail-open on a broken redactor, structlog nesting/non-string behavior, and the Lambda handler against synthetic gzip+base64 subscription events with stdout capture, a mocked Datadog urlopen, a mocked boto3 client, control-message and malformed-event handling, and one-bad-event batch survival). Rules-only Redactors throughout — no model download in tests.
- **Status:** built and tested locally only. NOT pushed, NOT committed — the push is Cory's call, per standing rule.

## 2026-10-27 — Rust SDK (crate phi-redact 0.6.0) + three-language perf comparison

- **Decision:** add a third SDK at `rust/` with feature parity to the 0.5.0 core surface (rules layer, merge pipeline rules > recognizers > neural, all five operators incl. the shared hash vectors, allow/deny, custom recognizers, per-label thresholds + needs-review, scrub/restore + ChunkedRestorer, redact_batch, a batch_jsonl example). All public offsets are CHARACTER indices (Python semantics); rule/denylist spans are scoreless, exactly like the other SDKs. The Python-only logging kit stays Python-only; no RemoteRedactor counterpart (callers inject spans).
- **Lookbehind re-implementation (chosen):** the Rust `regex` crate has no lookahead/lookbehind/backreferences. The phone pattern's `(?<=\w)\(` / `(?<![\w])` prefix and the denylist's `(?<!\w)term(?!\w)` are re-implemented as body patterns plus manual neighbour checks with finditer-compatible advancement (a boundary-rejected candidate advances one character). Equivalence is pinned by the ported Python tests (paren glued to a word char is consumed; a number glued to a word is rejected) and by the byte-parity harness.
- **Compiled-pattern cache (chosen, measured):** Python "recompiles" recognizer/denylist patterns per call but pays once via CPython's `re` cache. `regex::Regex::new` costs ~0.2 ms per pattern (Unicode tables); uncached, it put the Rust feature pipeline at 0.50 ms/note — slower than Python. The Rust engine compiles those patterns through a process-global cache (the `re`-cache analogue); feature median after the fix: 0.0127 ms/note. The built-in rules were already compile-once (LazyLock).
- **Neural via ort (verified, with a sandbox detour):** `src/neural.rs` ports OnnxEngine (chunking + 96-char overlap + degenerate-split guard, tokenizer truncation strip, vocab-derived marker ids, declared-inputs-only feeding, BIOES decode, f64 softmax). `ort-sys` 2.0.0-rc.10's prebuilt-binary download (cdn.pyke.io) returns HTTP 403 through the sandbox egress proxy — deterministic, and it fires even under `load-dynamic`. Working combination: `ORT_SKIP_DOWNLOAD=1` + `ort/load-dynamic` + `ORT_DYLIB_PATH` at the Python venv's onnxruntime 1.30 `libonnxruntime.so` (ort binds the 1.22 C API, which 1.30 serves). Verified against the reference: `rust/tests/neural.rs` pins the Python engine's predicted spans for three fixture notes on the real v9 int8 model — spans, texts, and rounded scores all match.
- **Perf (measured 2026-10-27, sandbox VM, 2× EPYC 9D25 vCPUs; full method + caveats in rust/PERF.md, raw numbers in rust/perf-results.json + bench/results.json):** per-note median — rules-only: Python 0.0538 ms, TS 0.0275, Rust 0.0071; full feature pipeline: 0.0711 / 0.0417 / 0.0127; batch over a 1,000-doc fixture-derived corpus: 13,708 / 45,407 / 98,530 notes/sec; full neural (v9 int8, fixture notes): Python 1.5256 ms vs Rust 1.6330 (a tie — both spend their time in ONNX Runtime 1.30), model load 0.728 s vs 0.327 s. TS neural is n/a (no local engine; RemoteRedactor only) and is reported as such, not estimated.
- **Evidence:** cargo test 104 passed (ported suites: validators, golden spans, phone boundaries, merge precedence, hash vectors, mask, allow/deny, recognizers, thresholds/review edges, restore, chunked split points) + 2 neural tests under `--features neural`; cross-SDK parity e2e 4 passed (TS basic/features + Rust basic/features — 16 feature configs + 2 batch configs × 15 notes byte-identical vs Python, now incl. scores and review lists).
- **Status:** built and tested locally only. NOT pushed, NOT committed — the push is Cory's call, per standing rule.

## 2026-10-27 — Benchmark audit + stage-decomposed re-bench (bench format v2)

- **Decision:** treat the three-language perf comparison as unproven until the harness itself is audited, then re-measure with stage decomposition. The audit (reading every bench file before trusting any number) found three defects in the first-pass sweep whose numbers appear in the Rust SDK entry above: (1) the mode labeled "rules-only" actually ran the full default redact path (scan + merge + operators) in all three harnesses — the old rules-only row is really the local pipeline; (2) no harness verified that the languages produced the same outputs on the bench inputs, so the comparison rested on assumed parity; (3) languages ran in one fixed sequential order (Python → TypeScript → Rust), letting machine drift systematically favor one end. Corpus and feature-config identity were verified sound (same two files everywhere; the feature config literal is identical across harnesses).
- **V2 protocol (chosen):** modes split into (a) raw rules scan, (b) local pipeline, (c) full feature pipeline, (d) batch for (b) and (c), (e) neural for Python/Rust with tokenize / ONNX inference / post-processing buckets measured inside the real predict path (Python: timing wrappers swapped onto OnnxEngine; Rust: new `OnnxEngine::predict_timed`, predict refactored through a shared inner fn). Every runner emits SHA-256 digests of a canonical output serialization (redacted text, spans with label/score/source/placeholder, review candidates) plus a digest of the inputs it actually parsed; the driver (`bench/run_bench.py`) interleaves 7 rounds (5 for batch) with rotating language order, pools raw samples, aggregates centrally, preserves per-round medians, and fails the run on any digest mismatch. `bench/results.json` format is now v2 (old aggregate fields retained under the same keys; stages/digests/per-rep added). Two harness bugs were found and fixed during the re-bench: batch digests were tied to the warm-up pass (absent when BENCH_WARMUPS=0), and per-note digests were computed after timing, so round 0 of each fresh process could include cold pattern-cache costs — the untimed digest pass now runs first in every invocation.
- **Parity verdict:** no engine divergence found. All output digests matched across all languages in all seven modes — scan, local, features, batch, batch-features, neural, neural-stages — including the scored digest of the full neural pipeline (Python and Rust produce identical spans with identical scores to 4 decimals on the real v9 model) and identical batch outputs over all 1,000 documents. Input digests matched everywhere; the seeded batch corpus regenerates byte-identical (SHA-256 c7ec2f1101f2…).
- **Results (supersede the perf paragraph of the Rust SDK entry above; canonical sweep in bench/results.json, mirrored to rust/perf-results.json):** per-note medians — (a) scan: Python 0.0483 ms, TS 0.0270, Rust 0.0070; (b) local pipeline: 0.0572 / 0.0313 / 0.0081; (c) features: 0.0726 / 0.0596 / 0.0129; (e) neural combined: Python 1.5691 vs Rust 1.2530. Batch: (b) 13,158 / 50,000 / 96,154 notes/sec; (c) 10,776 / 24,272 / 58,140. Neural decomposition: inference is 86% (Python) / 90% (Rust) of predict() — both bind ONNX Runtime 1.30, so the neural rows are a tie within noise; the host-language difference is post-processing (0.121 vs 0.041 ms) and tokenize (~0.1 ms both). Repeatability: the full sweep ran twice back-to-back; local-stage medians reproduced within ~7% except TS scan (19%), TS batch throughput varied ~20%, neural p95s are outlier-driven while medians held within ~12%.
- **Docs:** PERFORMANCE.md (root) and rust/PERF.md rewritten: apples-to-apples verification section (corpus hashes, config identity, digest-equality results, timed-region definitions, inherent per-platform asymmetries stated as facts — TS fresh RegExp per scan, Python re-cache, Rust compiled-pattern cache, offset semantics), stage tables, rules-vs-neural summary, and a choosing-a-language guide grounded in the measurements plus real feature differences (Python: only SDK with the logging kit + local neural; TS: browser/Node, neural via server; Rust: highest throughput + local neural, no logging kit). Terminology per Cory: the neural layer is a "neural (RoBERTa-style token classifier)", not an LLM.
- **Status:** working tree only. NOT pushed, NOT committed — the push is Cory's call, per standing rule.
