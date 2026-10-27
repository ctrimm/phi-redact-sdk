# phi-redact TypeScript SDK

```ts
import { redact, restore, scan } from "phi-redact";

const r = redact("Call Jane Doe at 415-555-0132");
r.redacted; // 'Call Jane Doe at [PHONE_1]'
r.mapping; // { '[PHONE_1]': '415-555-0132' }
restore(r.redacted, r.mapping); // original back
```

Zero dependencies. Rules-only, synchronous, works in Node 18+ and browsers.

## Install

```bash
npm install phi-redact
# or from this repo:
npm --prefix typescript install && npm --prefix typescript run build
```

## API

**Local engine** (deterministic rules layer — cards, SSN, IBAN, phone,
email, URL, IP, ZIP, date, MRN; checksum-gated):

- `scan(text)` -> `Entity[]` (`start`, `end`, `label`, `text`, `source`)
- `redact(text, neuralEntities?, options?)` -> `RedactionResult`
  (`redacted`, `entities`, `mapping`, `counts`, `engine`, `model`);
  `options` carries `operator`, `salt`, `allowlist`, `denylist`,
  `denyLabel` (see below)
- `redactBatch(texts, options?)` -> one `RedactionResult` per input,
  in order
- `scrub(text, neuralEntities?, options?)` -> `{ redacted, mapping }`
  for the LLM round-trip; `restore(redacted, mapping)` -> original text
- `ChunkedRestorer` — streaming restore for chunked LLM output
- `mergeEntities(ruleEnts, neuralEnts?)` -> merged list; rules win every
  overlap, neural-neural overlaps keep the higher `score`
- `applyPlaceholders(text, entities, options?)` -> `{ redacted, mapping }`
- Validators exported individually: `luhnOk`, `ssnValid`, `ibanValid`

**Remote client** (rules + ONNX neural layer via `redact serve`):

```ts
import { RemoteRedactor } from "phi-redact";

const rr = new RemoteRedactor("http://127.0.0.1:8080");
await rr.health();
const r = await rr.redact(text, 0.5); // optional per-call threshold
await rr.restore(r.redacted, r.mapping);
```

Start the server from the Python side first:

```bash
serve/redact serve --port 8080 --model model.onnx --tokenizer tokenizer.json
```

`RemoteRedactor` accepts `{ baseUrl, fetchFn }` (or a plain base-URL
string); `fetchFn` lets you inject a custom fetch for tests or
non-Node runtimes.

## Placeholders

Numbered per label in document order: `[NAME_PATIENT_1]`, `[PHONE_2]`,
... Keep the mapping for pseudonymized round-trips; delete it and the
copy is anonymized. Matches the Python SDK's numbering exactly.

## Batch pipelines

`redactBatch(texts, options?)` redacts many texts in one call, results
in input order. For files, `examples/batch_jsonl.mjs` is a
dependency-free JSONL processor (records in, redacted records plus a
`counts` object out, stdin/file to stdout/file):

```bash
node examples/batch_jsonl.mjs notes.jsonl redacted.jsonl
cat notes.jsonl | node examples/batch_jsonl.mjs --operator hash --salt "$SALT"
```

## Scrub & restore (LLM round-trips)

Scrub a prompt before it leaves for a cloud LLM; restore the response
locally:

```ts
const { redacted, mapping } = scrub(prompt);   // send `redacted` out
const answer = await callCloudLlm(redacted);
const restored = restore(answer, mapping);
```

If the response streams, `ChunkedRestorer` restores placeholders split
across chunks and passes unknown/partial placeholders through
untouched:

```ts
const cr = new ChunkedRestorer(mapping);
for await (const chunk of stream) emit(cr.feed(chunk));
emit(cr.flush());
```

Round-trips require the default `numbered` operator (see below).

## Anonymize operators

The `operator` option controls how spans are rendered:

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
dictionary-attackable. (SHA-256 is implemented in pure TS — the SDK
stays zero-dependency and browser-safe.)

## Allow / deny lists

`redact` options:

- `allowlist: [...]` — terms that are never redacted. A detected span
  whose text matches a term (case-insensitive, exact) is dropped.
- `denylist: [...]` — terms that are always redacted. Exact matches
  (case-insensitive, word-boundary aware, so `Codral` does not match
  inside `Codraline`) are injected as spans labeled `denyLabel`
  (default `"CUSTOM"`). A deny span wins any overlap with a detected
  span, and deny wins over allow when a term is in both lists.

## Presidio-parity use cases

0.4.0 closes the SDK-layer gaps found in the Presidio comparison
(`phi-redact/evals/presidio-vs-phiredact/`), for English text:

- **Enterprise batch pipelines** — `redactBatch` plus the
  dependency-free JSONL processor in `examples/`.
- **Scrubbing prompts before cloud LLM calls** — the `scrub` /
  `restore` pair and `ChunkedRestorer` for streaming responses.
- **Anonymization options** — Presidio-style operators (`label`,
  `mask`, `hash`, `remove`) alongside the default numbered
  placeholders, with cross-SDK-identical hash tokens for
  join-preserving analytics.
- **Allow/deny lists** — site-specific exceptions and always-redact
  terms without writing custom recognizers.

Explicitly out of scope: image redaction (OCR/DICOM), structured-data
anonymization, and non-English / international recognizers — Presidio
remains the right tool for those lanes. The local engine is rules-only;
the neural layer runs through `RemoteRedactor` (batch and operator
options apply to the local engine).

## Scope note

This SDK redacts text. It is not a certified de-identification solution
and makes no HIPAA Expert Determination / Safe Harbor claim.

## Tests

```bash
npm test   # builds, then runs the unit + RemoteRedactor + 0.4.0 feature suites (69 tests)
```
