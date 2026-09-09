#!/usr/bin/env bash
set -euo pipefail

"${PYTHON:-.venv/bin/python}" -m megatron_lab.inspect_recipe "$@"
