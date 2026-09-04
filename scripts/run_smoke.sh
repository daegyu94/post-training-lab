#!/usr/bin/env bash
set -euo pipefail

if [[ ! -x .venv/bin/python ]]; then
    echo "Run ./scripts/setup.sh first." >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
.venv/bin/python -m sft_lab.train \
    --train-samples "${TRAIN_SAMPLES:-32}" \
    --eval-samples "${EVAL_SAMPLES:-8}" \
    --max-steps "${MAX_STEPS:-5}" \
    --output-dir "${OUTPUT_DIR:-results/qwen2.5-14b-qlora-smoke}" \
    "$@"
