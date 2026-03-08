#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Check if port 8000 is already in use
if lsof -ti:8000 >/dev/null 2>&1; then
    echo "Port 8000 is already in use. Run ./stop_demo.sh first."
    exit 1
fi

echo "=== Step 1: Installing dependencies ==="
uv sync

echo
echo "=== Step 2: Building Rust extension ==="
uv pip install maturin
uv run maturin develop --manifest-path multiverse_provider/Cargo.toml

echo
echo "=== Step 3: Starting web app (background) ==="
PYTHONPATH="src:src/app" uv run uvicorn server:app --host 0.0.0.0 --port 8000 &
APP_PID=$!
sleep 2
echo "Web app running at http://localhost:8000 (PID $APP_PID)"

echo
echo "=== Step 4: Launching pipelines ==="
uv run python src/demo.py "$@"

echo
echo "All pipelines finished. App is still running at http://localhost:8000."
echo "Run ./stop_demo.sh to stop."
wait $APP_PID
