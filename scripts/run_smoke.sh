#!/usr/bin/env bash
set -euo pipefail

if [[ ! -x .venv/bin/python ]]; then
    echo "Run ./scripts/setup.sh first." >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
dataset_parquet_dir="${DATASET_PARQUET_DIR:-data/ultrachat_200k/data}"

if [[ ! -d "$dataset_parquet_dir" ]]; then
    echo "Dataset directory not found: $dataset_parquet_dir" >&2
    echo "Run ./scripts/setup.sh first." >&2
    exit 1
fi

.venv/bin/python -m sft_lab.train \
    --train-samples "${TRAIN_SAMPLES:-32}" \
    --eval-samples "${EVAL_SAMPLES:-8}" \
    --max-steps "${MAX_STEPS:-5}" \
    --output-dir "${OUTPUT_DIR:-results/qwen2.5-14b-qlora-smoke}" \
    "$@" \
    --dataset-parquet-dir "$dataset_parquet_dir" \
    --local-files-only
