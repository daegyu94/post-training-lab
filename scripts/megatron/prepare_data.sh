#!/usr/bin/env bash
set -euo pipefail

venv_dir="${VENV_DIR:-.venv-megatron}"
"$venv_dir/bin/python" -m megatron_lab.prepare_data \
  --train-samples "${TRAIN_SAMPLES:-32}" \
  --eval-samples "${EVAL_SAMPLES:-8}" \
  --seed "${SEED:-42}" \
  --output-dir "${DATA_DIR:-data/ultrachat_200k}"
