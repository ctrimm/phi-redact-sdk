#!/usr/bin/env bash
# Reproduce the three-SDK perf comparison (see PERFORMANCE.md and
# rust/PERF.md). Builds the Rust bench binary (neural feature), then
# runs bench/run_bench.py, which interleaves the three harnesses
# round-by-round, verifies corpus/output digests across languages,
# and writes bench/results.json (format v2).
#
#   bench/run_bench.sh            # full sweep incl. neural modes
#   NEURAL=0 bench/run_bench.sh   # local pipeline only
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
export PATH="$HOME/workspace/tooling/rust/cargo/bin:$PATH"
export RUSTUP_HOME="${RUSTUP_HOME:-$HOME/workspace/tooling/rust/rustup}"
export CARGO_HOME="${CARGO_HOME:-$HOME/workspace/tooling/rust/cargo}"

python3 bench/gen_corpus.py > bench/corpus.jsonl
(cd rust && cargo build --release --features neural --example bench 2>/dev/null)

NEURAL="${NEURAL:-1}" python3 bench/run_bench.py
echo "wrote $ROOT/bench/results.json"
