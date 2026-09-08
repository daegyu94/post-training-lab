#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${DATASET_REVISION:-}" ]]; then
  echo "DATASET_REVISION must be a Hugging Face commit SHA" >&2
  exit 2
fi

python_bin="${PYTHON:-.venv/bin/python}"
"$python_bin" -m megatron_lab.prepare_data \
  --train-samples "${TRAIN_SAMPLES:-32}" \
  --eval-samples "${EVAL_SAMPLES:-8}" \
  --seed "${SEED:-42}" \
  --dataset-revision "$DATASET_REVISION" \
  --output-dir "${DATA_DIR:-data/ultrachat_200k}"
