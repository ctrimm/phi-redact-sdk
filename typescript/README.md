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
- `redact(text, neuralEntities?)` -> `RedactionResult` (`redacted`,
  `entities`, `mapping`, `counts`, `engine`, `model`)
- `restore(redacted, mapping)` -> original text
- `mergeEntities(ruleEnts, neuralEnts?)` -> merged list; rules win every
  overlap, neural-neural overlaps keep the higher `score`
- `applyPlaceholders(text, entities)` -> `{ redacted, mapping }`
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

## Scope note

This SDK redacts text. It is not a certified de-identification solution
and makes no HIPAA Expert Determination / Safe Harbor claim.

## Tests

```bash
npm test   # builds, then runs the unit + RemoteRedactor integration suites
```
