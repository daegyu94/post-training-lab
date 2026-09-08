#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON:-.venv/bin/python}"
"$python_bin" -m megatron_lab.prepare_public_data "$@"
