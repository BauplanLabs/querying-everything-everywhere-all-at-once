#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Installing Python dependencies ==="
uv sync

echo
echo "=== Building Rust extension ==="
uv pip install maturin
uv run maturin develop --manifest-path multiverse_provider/Cargo.toml

echo
echo "=== Running benchmarks ==="
uv run python src/benchmarks/bench.py "$@"
