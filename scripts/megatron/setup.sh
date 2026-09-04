#!/usr/bin/env bash
set -euo pipefail

venv_dir="${VENV_DIR:-.venv-megatron}"
if [[ ! -x "$venv_dir/bin/python" ]]; then
  python -m venv "$venv_dir"
fi

. "$venv_dir/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r requirements/megatron.txt
