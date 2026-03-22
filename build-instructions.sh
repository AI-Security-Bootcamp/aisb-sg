#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [ -n "${VIRTUAL_ENV:-}" ]; then
    echo "Using activated virtual environment at '$VIRTUAL_ENV'"
elif [ -d ".venv" ]; then
    echo "Using existing virtual environment in '.venv/'"
    source .venv/bin/activate
else
    echo "Creating a virtual environment..."
    python -m venv .venv
    echo "Virtual environment created in 'venv/'"
    source .venv/bin/activate
    pip install -r requirements.txt
fi

python aisb_utils/build_instructions.py "$@"