#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../../.." && pwd)"
python_bin="${PYTHON:-python3}"
export PYTHONPATH="$repo_root/backends/verl${PYTHONPATH:+:$PYTHONPATH}"

exec "$python_bin" -m verl_lab.preflight "$@"
