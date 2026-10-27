# E2E: cross-SDK parity

The Python, TypeScript, and Rust SDKs are independent ports of the
same rules layer. This harness keeps them honest: the same synthetic
notes go through all of them, and redacted text, entity spans, and
mappings must match exactly.

## Run

```bash
cd typescript && npm install && npm run build && cd ..
python3 -m pytest e2e/parity/ -q
```

The pytest run shells out to `node e2e/parity/run_ts.mjs` for the
TypeScript side and runs `phi_redact` in-process for the Python side.
It skips (does not fail) when `node` is missing or `typescript/dist`
has not been built. `test_parity_rust.py` does the same for the Rust
SDK via `rust/examples/parity_runner.rs` (built in release mode on
first use; skips when cargo is missing).

## Fixtures

`notes.jsonl` — one JSON object per line: `{"note": "..."}`. Synthetic
data only; never put real PII here. Add a line when a new entity type or
edge case needs parity coverage.
