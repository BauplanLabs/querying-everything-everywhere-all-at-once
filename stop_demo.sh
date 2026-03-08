#!/usr/bin/env bash
set -euo pipefail

echo "Stopping demo app on port 8000..."
PIDS=$(lsof -ti:8000 2>/dev/null || true)
if [ -z "$PIDS" ]; then
    echo "Nothing running on port 8000."
else
    echo "$PIDS" | xargs kill -9 2>/dev/null || true
    echo "Killed."
fi
