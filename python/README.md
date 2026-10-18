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
- `phi_redact.redact(text)` -> `RedactionResult` (`redacted`, `entities`,
  `mapping`, `counts`, `engine`, `model`)
- `phi_redact.restore(redacted, mapping)` -> original text
- `RedactionResult.to_dict()` -> JSON-serializable dict

For the neural layer (names, addresses, free-text PHI the rules cannot
see), use `Redactor`:

```python
r = phi_redact.Redactor(model="model.onnx", tokenizer="tokenizer.json")
result = r.redact(text, threshold=0.5)   # per-call threshold override
r.scan(text)                             # merged entity list, no redaction
```

`Redactor(model=...)` loads the ONNX model once and reuses the session.
`labels` optionally points at a `config.json` carrying `id2label`
(default: `config.json` next to the model).

Export a checkpoint first:

```bash
.venv/bin/python train/export_onnx.py --model train/checkpoint-best \
    --tokenizer train/tokenizer-full/tokenizer.json --out model.onnx
```

## Merge policy

Rule spans always win overlaps (checksum-gated precision). Neural spans
that overlap a rule span are dropped; neural-neural overlaps keep the
higher confidence. Placeholders are numbered per label in document order:
`[NAME_PATIENT_1]`, `[PHONE_2]`, ... Keep the mapping for pseudonymized
round-trips; delete it and the copy is anonymized.

## Scope note

This SDK redacts text. It is not a certified de-identification solution
and makes no HIPAA Expert Determination / Safe Harbor claim.
