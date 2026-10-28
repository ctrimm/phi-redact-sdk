#!/usr/bin/env bash
# Reproduce the four-SDK perf comparison (see PERFORMANCE.md and
# rust/PERF.md). Builds the Rust bench binary (neural feature) and
# the Swift bench binary (release, scratch path outside the repo),
# then runs bench/run_bench.py, which interleaves the four
# harnesses round-by-round, verifies corpus/output digests across
# languages, and writes bench/results.json (format v2).
#
#   bench/run_bench.sh            # full sweep incl. neural modes
#   NEURAL=0 bench/run_bench.sh   # local pipeline only
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
export PATH="$HOME/workspace/tooling/rust/cargo/bin:$PATH"
export RUSTUP_HOME="${RUSTUP_HOME:-$HOME/workspace/tooling/rust/rustup}"
export CARGO_HOME="${CARGO_HOME:-$HOME/workspace/tooling/rust/cargo}"
# ort-sys downloads its prebuilt ONNX Runtime from a CDN that is
# unreachable from this sandbox (403 via the egress proxy); the
# working combination here is skip-download + load-dynamic, with
# the driver pointing ORT_DYLIB_PATH at the Python wheel's runtime.
# Override with ORT_SKIP_DOWNLOAD=0 where the CDN is reachable.
export ORT_SKIP_DOWNLOAD="${ORT_SKIP_DOWNLOAD:-1}"
SWIFT_BIN="${SWIFT_BIN:-$HOME/workspace/tooling/swift-6.1/toolchain/usr/bin/swift}"
export SW_BENCH_SCRATCH="${SW_BENCH_SCRATCH:-/tmp/phi-bench-swift}"

python3 bench/gen_corpus.py > bench/corpus.jsonl
(cd rust && cargo build --release --features neural --example bench 2>/dev/null)
"$SWIFT_BIN" build -c release --package-path bench/swift \
    --scratch-path "$SW_BENCH_SCRATCH" 2>/dev/null

NEURAL="${NEURAL:-1}" python3 bench/run_bench.py
echo "wrote $ROOT/bench/results.json"
