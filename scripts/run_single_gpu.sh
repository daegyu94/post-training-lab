#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  .venv/bin/torchrun --standalone --nproc-per-node=1 \
  -m megatron_lab.train --tp 1 \
  --output-dir results/single-gpu "$@"
