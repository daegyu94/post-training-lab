#!/usr/bin/env bash
set -euo pipefail

if [[ ! -x .venv/bin/python ]]; then
    echo "Run ./scripts/setup.sh first." >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
dataset_parquet_dir="${DATASET_PARQUET_DIR:-data/ultrachat_200k/data}"
output_dir="${OUTPUT_DIR:-results/trl-branch-experiment}"

if [[ ! -d "$dataset_parquet_dir" ]]; then
    echo "Dataset directory not found: $dataset_parquet_dir" >&2
    echo "Run ./scripts/setup.sh first or set DATASET_PARQUET_DIR." >&2
    exit 1
fi

HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    .venv/bin/python -m sft_lab.train \
    --dataset-parquet-dir "$dataset_parquet_dir" \
    --local-files-only \
    --train-samples 128 \
    --eval-samples 16 \
    --max-steps 20 \
    --max-length 512 \
    --gradient-accumulation-steps 8 \
    --output-dir "$output_dir" \
    "$@"
