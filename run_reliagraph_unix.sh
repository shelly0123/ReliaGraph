#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ ! -x .venv/bin/python ]]; then
  "$PYTHON_BIN" -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dashboard.txt
echo "ReliaGraph is starting at http://127.0.0.1:8765 (local only). Press Ctrl+C to stop."
python -m uvicorn webapp.main:app --host 127.0.0.1 --port 8765
