#!/usr/bin/env bash
# Reproduce the three-SDK perf comparison (see rust/PERF.md).
# Runs every bench harness sequentially on this machine and writes
# bench/results.json. Rust runs from rust/ (paths are relative there);
# set NEURAL=1 to include the neural modes (needs the v9 model assets
# and, for Rust, a --features neural build).
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
VENV_PY="${VENV_PY:-$HOME/workspace/phi-redact/.venv/bin/python}"
MODEL="${MODEL:-$HOME/workspace/phi-redact/work/v9/phi-redact-v9-int8.onnx}"
TOK="${TOK:-$HOME/workspace/phi-redact/work/v9/mac-model-dir/tokenizer.json}"
TAGS="${TAGS:-$HOME/workspace/phi-redact/work/v9/mac-model-dir/tags.json}"

python3 bench/gen_corpus.py > bench/corpus.jsonl
# Python's OnnxEngine wants a config.json with id2label; derive one
# from the v9 tags.json array into a scratch dir.
ASSETS="$(mktemp -d)"
python3 - "$TAGS" "$ASSETS/config.json" <<'EOF'
import json, sys
tags = json.load(open(sys.argv[1]))
json.dump({"id2label": {str(i): t for i, t in enumerate(tags)},
           "label2id": {t: i for i, t in enumerate(tags)}},
          open(sys.argv[2], "w"))
EOF

OUT="$ROOT/bench/results.json"
{
  echo '{'
  echo "  \"environment\": {"
  echo "    \"python\": \"$($VENV_PY --version 2>&1)\","
  echo "    \"node\": \"$(node --version)\","
  echo "    \"rustc\": \"$(rustc --version)\","
  echo "    \"cpu\": \"$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)\","
  echo "    \"cpus\": $(nproc),"
  echo "    \"os\": \"$(uname -sr)\""
  echo '  },'
  echo '  "runs": ['
} > "$OUT"

first=1
emit() {  # $1 = one JSON line from a bench harness
  if [ $first -eq 1 ]; then first=0; else echo ',' >> "$OUT"; fi
  printf '    %s' "$1" >> "$OUT"
}

export PYTHONPATH="$ROOT/python"
emit "$($VENV_PY bench/bench_python.py rules)"
emit "$($VENV_PY bench/bench_python.py features)"
emit "$(node bench/bench_ts.mjs rules)"
emit "$(node bench/bench_ts.mjs features)"
emit "$( (cd rust && ./target/release/examples/bench rules) )"
emit "$( (cd rust && ./target/release/examples/bench features) )"
emit "$($VENV_PY bench/bench_python.py batch)"
emit "$(node bench/bench_ts.mjs batch)"
emit "$( (cd rust && ./target/release/examples/bench batch) )"
if [ "${NEURAL:-0}" = "1" ]; then
  emit "$($VENV_PY bench/bench_python.py neural "$MODEL" "$TOK" "$ASSETS/config.json")"
  # Rust neural: requires the bench example built with the feature —
  #   (cd rust && cargo build --release --features neural --example bench)
  # which replaces target/release/examples/bench with the neural build.
  emit "$( (cd rust && ./target/release/examples/bench neural "$MODEL" "$TOK" "$TAGS") )"
fi

{
  echo ''
  echo '  ]'
  echo '}'
} >> "$OUT"
echo "wrote $OUT"
