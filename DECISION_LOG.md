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
