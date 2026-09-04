#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
  CUDA_DEVICE_MAX_CONNECTIONS=1 \
  .venv/bin/torchrun --standalone --nproc-per-node=2 \
  -m megatron_lab.train --tp 2 \
  --output-dir results/two-gpu-tp "$@"
