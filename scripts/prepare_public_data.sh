#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

python_bin="${PYTHON:?Set PYTHON to a backend venv interpreter}"
"$python_bin" -m datasets_lab.prepare_public_data "$@"
