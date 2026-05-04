#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
MODE="${1:-cpu}"

if [[ "$MODE" != "cpu" && "$MODE" != "--cpu" && "$MODE" != "cuda" && "$MODE" != "--cuda" ]]; then
    echo "usage: $0 [cpu|cuda]" >&2
    exit 2
fi

python -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip

if [[ "$MODE" == "cuda" || "$MODE" == "--cuda" ]]; then
    "$VENV_DIR/bin/python" -m pip install -r "$ROOT_DIR/requirements.txt"
else
    "$VENV_DIR/bin/python" -m pip install -r "$ROOT_DIR/requirements-cpu.txt"
fi

echo "Environment ready: $VENV_DIR"
echo "Activate with: source $VENV_DIR/bin/activate"
