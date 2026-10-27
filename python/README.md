# phi-redact Python SDK

```python
import phi_redact

result = phi_redact.redact("Call Jane Doe at 415-555-0132")
result.redacted   # 'Call Jane Doe at [PHONE_1]'
result.mapping    # {'[PHONE_1]': '415-555-0132'}
phi_redact.restore(result.redacted, result.mapping)
```

## Install

```bash
pip install ./python            # rules-only, zero dependencies
pip install "./python[onnx]"    # plus the ONNX neural layer
# or straight from GitHub:
pip install "git+https://github.com/ctrimm/phi-redact-sdk#subdirectory=python"
```

## API

Module-level functions are rules-only (checksum-gated scans for cards,
SSN, IBAN, phone, email, URL, IP, ZIP, date, MRN):

- `phi_redact.scan(text)` -> entity dicts (`start`, `end`, `label`,
  `text`, `source`), sorted by start
- `phi_redact.redact(text, operator=..., salt=..., allowlist=...,
  denylist=..., deny_label=..., recognizers=..., thresholds=...,
  review_margin=...)` -> `RedactionResult` (`redacted`, `entities`,
  `mapping`, `counts`, `engine`, `model`, `review`); the keyword
  options mirror the `Redactor` constructor options below
- `phi_redact.restore(redacted, mapping)` -> original text
- `RedactionResult.to_dict()` -> JSON-serializable dict

For the neural layer (names, addresses, free-text PHI the rules cannot
see), use `Redactor`:

```python
r = phi_redact.Redactor(model="model.onnx", tokenizer="tokenizer.json")
result = r.redact(text, threshold=0.5)   # per-call threshold override
r.scan(text)                             # merged entity list, no redaction
```

The current model is **v9** (2026-10-21) — 129 tags / 32 entity types,
11.3 MB INT8 ONNX:

```bash
huggingface-cli download CDT5058/phi-redact \
    phi-redact-v9-int8.onnx tokenizer.json config.json
```

```python
r = phi_redact.Redactor(model="phi-redact-v9-int8.onnx",
                        tokenizer="tokenizer.json")  # labels: config.json beside the model
```

Golden-set scores (rules + neural, default threshold 0.6): full-value
recall 0.974, term recall 0.828, precision 0.746, clean-case FP rate
0.200. v8 (`phi-redact-v8-int8.onnx`) and v7 (`phi-redact-v7-int8.onnx`,
121 tags) remain in the same repo.

`Redactor(model=...)` loads the ONNX model once and reuses the session.
`labels` optionally points at a `config.json` carrying `id2label`
(default: `config.json` next to the model).

Export a checkpoint first:

```bash
.venv/bin/python train/export_onnx.py --model train/checkpoint-best \
    --tokenizer train/tokenizer-full/tokenizer.json --out model.onnx
```

## Merge policy

Precedence is rules > recognizers > neural. Rule spans win every
overlap (checksum-gated precision); recognizer spans beat neural
spans but yield to rules; neural spans that overlap a kept span are
dropped, and neural-neural overlaps keep the higher confidence.
Placeholders are numbered per label in document order:
`[NAME_PATIENT_1]`, `[PHONE_2]`, ... Keep the mapping for pseudonymized
round-trips; delete it and the copy is anonymized.

## Batch pipelines

One loaded instance, many documents:

```python
r = phi_redact.Redactor(model="phi-redact-v9-int8.onnx",
                        tokenizer="tokenizer.json")
results = r.redact_batch(texts)        # list, input order preserved
for result in r.redact_iter(texts):    # generator: streams, low memory
    ...
```

For files, `examples/batch_jsonl.py` is a dependency-free JSONL
processor: records in, redacted records (text field replaced, plus a
`counts` object) out, from stdin/file to stdout/file.

```bash
python3 examples/batch_jsonl.py notes.jsonl redacted.jsonl
cat notes.jsonl | python3 examples/batch_jsonl.py --operator hash --salt "$SALT"
```

## Scrub & restore (LLM round-trips)

Scrub a prompt before it leaves for a cloud LLM; restore the response
locally. Works with the neural layer too, not just the rules:

```python
redacted, mapping = r.scrub(prompt)        # send `redacted` out
answer = call_cloud_llm(redacted)
restored = phi_redact.Redactor.restore(answer, mapping)
```

If the response streams, `ChunkedRestorer` restores placeholders that
are split across chunks and passes unknown/partial placeholders through
untouched:

```python
cr = phi_redact.ChunkedRestorer(mapping)
for chunk in stream:
    emit(cr.feed(chunk))
emit(cr.flush())
```

Round-trips require the default `numbered` operator (see below).

## Anonymize operators

The `operator` option (Redactor constructor or `redact()` keyword)
controls how spans are rendered:

| operator | rendering | mapping | use |
|---|---|---|---|
| `numbered` (default) | `[PHONE_1]` | populated | reversible pseudonymization |
| `label` | `<PHONE>` (Presidio-style) | empty | irreversible, human-readable |
| `mask` | `***-**-7890` | empty | partial reveal for support workflows |
| `hash` | `[SSN_e2e9f611]` | empty | consistent pseudonyms, join-preserving |
| `remove` | *(deleted)* | empty | hard deletion |

Only `numbered` is reversible; the others return an empty mapping by
design. Masking keeps the last 4 characters for structured IDs (SSN,
card, phone, and kin) of length >= 8 with separators preserved; every
other value is masked to its own length. Hash tokens are the first 8
hex chars of `SHA-256(salt \0 label \0 value)` — the same value and
salt produce the same token in every document and in both SDKs, so
redacted datasets still join. **Use a secret salt**: unsalted hashes of
low-entropy values (an SSN has ~1e9 possibilities) are
dictionary-attackable.

## Allow / deny lists

Redactor options (also `redact()` keywords):

- `allowlist=[...]` — terms that are never redacted. A detected span
  whose text matches a term (case-insensitive, exact) is dropped.
- `denylist=[...]` — terms that are always redacted. Exact matches
  (case-insensitive, word-boundary aware, so `Codral` does not match
  inside `Codraline`) are injected as spans labeled `deny_label`
  (default `"CUSTOM"`). A deny span wins any overlap with a detected
  span, and deny wins over allow when a term is in both lists.

## Custom recognizers

Site-specific identifier formats the built-in layers do not know —
employee IDs, internal account numbers, case codes — are a
`PatternRecognizer` (Redactor option or `redact()` keyword):

```python
from phi_redact import PatternRecognizer

emp = PatternRecognizer(
    label="EMPLOYEE_ID",                 # custom labels allowed
    patterns=[r"EMP\d{6}"],              # one or more regexes
    score=0.85,                          # base confidence
    context=["badge", "employee"],       # optional context words
    context_window=40,                   # chars searched per side
    context_boost=0.10,                  # score bump on a context hit
    require_context=False,               # True: no context, no match
    validator=None,                      # callable, or "luhn"/"ssn"/
                                         # "iban"/"aba" checksum name
    group=0,                             # match group used as the span
)
r = phi_redact.Redactor(recognizers=[emp])
```

Recognizer spans carry `source="recognizer"` and a score, and merge
under a three-tier precedence: **rules > recognizers > neural** (the
built-in rules are effectively registered first). Among recognizers,
the first registered wins an overlap. Recognizer matches respect the
allowlist (a protected value is dropped) and lose overlaps to the
denylist (deny still wins). They redact under whichever anonymize
operator is configured, like any other span. Patterns should stick to
the regex subset shared by Python `re` and JS `RegExp` if the same
recognizers are used with the TypeScript SDK.

## Per-label thresholds & needs-review mode

Scored spans (neural and recognizer spans) are gated per label:

```python
r = phi_redact.Redactor(
    thresholds={"NAME": 0.75, "EMPLOYEE_ID": 0.9},  # label -> gate
    review_margin=0.15,
)
result = r.redact(text)
result.review   # [{'start': .., 'end': .., 'label': .., 'text': ..,
                #   'score': .., 'source': ..}, ...]
```

A label's threshold overrides the global one (`Redactor(threshold=)`,
default 0.6) for spans of that label. A span scoring at or above its
threshold redacts; below it, if `review_margin` is set and the score
is at or above `threshold - review_margin`, the span is **not**
redacted — it comes back on `result.review` for a human to decide
(the human-in-the-loop surface Presidio does not have). Anything
lower is dropped. Review candidates are never in the mapping or the
counts, and `restore()` ignores them (they were never replaced).
Batch results carry their own `review` list per result.

Deliberate edge: spans **without** a score — the checksum-gated rules
layer and denylist injections — are never gated and never reviewed.
A per-label threshold on `"SSN"` does not stop the SSN rule; thresholds
govern the probabilistic layers, rules stay deterministic.

## Logging integrations

Scrub log lines at emit time, so PII never reaches the log store —
CloudWatch, Datadog, or anything downstream of your logging setup.
Full walkthroughs (both vendors, both placements, TypeScript notes):
[docs/logging-integrations.md](docs/logging-integrations.md).

A stdlib `logging` filter renders each record, redacts it, and hands
the scrubbed line to your existing handlers:

```python
import logging
import phi_redact
from phi_redact.integrations.logging_kit import add_redacting_filter

r = phi_redact.Redactor(model="phi-redact-v9-int8.onnx",
                        tokenizer="tokenizer.json")
add_redacting_filter(logging.getLogger(), r)   # or pass a handler

logging.info("patient SSN 234-56-7890 checked in")
# emits: patient SSN [SSN_1] checked in
```

`RedactingFilter(redactor, scrub_args=True, scrub_exc=True)` is the
class behind it: `scrub_args` covers the rendered message (template +
`%`-args); `scrub_exc` also scrubs formatted exception text, so a
traceback mentioning an identifier is covered. The filter fails open —
a redaction error passes the original record through rather than
breaking the application's logging (see the module docstring for the
tradeoff).

For structlog, `make_structlog_processor(r)` returns a processor that
scrubs the event string and, recursively, string leaves of nested
dicts/lists. Dict keys are field names and are not scrubbed. structlog
itself is not imported — the processor is duck-typed.

For logs you cannot change at the source, `examples/lambda_log_scrubber.py`
is a CloudWatch Logs subscription-filter Lambda: it decodes each
batch, scrubs every event, and forwards to a destination picked by
`LOG_SCRUB_DEST`. Pipeline-side scrubbing protects downstream
systems; the raw lines still sit in the source log group — the guide
explains the placement tradeoff.

| Env var | Meaning |
|---|---|
| `LOG_SCRUB_DEST` | `stdout` (default) — the Lambda's own log group becomes the scrubbed store · `datadog` · `cloudwatch` |
| `DD_API_KEY` | Datadog API key (dest `datadog`) |
| `LOG_SCRUB_TARGET_LOG_GROUP` | clean log group to write (dest `cloudwatch`) |
| `PHI_REDACT_MODEL` / `PHI_REDACT_TOKENIZER` | optional ONNX paths; unset = rules-only |
| `PHI_REDACT_OPERATOR` / `PHI_REDACT_SALT` / `PHI_REDACT_THRESHOLD` | Redactor options |

## Presidio-parity use cases

0.4.0 closes the SDK-layer gaps found in the Presidio comparison
(`phi-redact/evals/presidio-vs-phiredact/`), for English text:

- **Enterprise batch pipelines** — `redact_batch` / `redact_iter` plus
  the dependency-free JSONL processors in `examples/`.
- **Scrubbing prompts before cloud LLM calls** — the `scrub` /
  `restore` pair on `Redactor` (neural pipeline included) and
  `ChunkedRestorer` for streaming responses.
- **Anonymization options** — Presidio-style operators (`label`,
  `mask`, `hash`, `remove`) alongside the default numbered
  placeholders, with cross-SDK-identical hash tokens for
  join-preserving analytics.
- **Allow/deny lists** — site-specific exceptions and always-redact
  terms without writing custom recognizers.

0.5.0 adds the next two:

- **Custom recognizers** — Presidio's PatternRecognizer, ported:
  user regexes with base scores, context-word boosting, and
  checksum/callable validators, under built-in or custom labels.
- **Per-label thresholds + needs-review** — Presidio-style
  per-entity score gates, plus a review band that returns
  borderline spans for human adjudication instead of dropping them.

Explicitly out of scope: image redaction (OCR/DICOM), structured-data
anonymization, and non-English / international recognizers — Presidio
remains the right tool for those lanes.

## Scope note

This SDK redacts text. It is not a certified de-identification solution
and makes no HIPAA Expert Determination / Safe Harbor claim.

## Tests

```bash
python3 -m pytest tests/ -q          # unit + integration (158 tests)
python3 -m pytest e2e/parity -q      # cross-SDK parity vs the TypeScript port
```
