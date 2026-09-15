#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../../.." && pwd)"
if [[ -n "${PYTHON:-}" ]]; then
  python_bin="$PYTHON"
elif [[ -x "$repo_root/.venv/bin/python" ]]; then
  python_bin="$repo_root/.venv/bin/python"
else
  python_bin="python3"
fi
export PYTHONPATH="$repo_root/backends/verl${PYTHONPATH:+:$PYTHONPATH}"

exec "$python_bin" -m verl_lab.preflight "$@"
