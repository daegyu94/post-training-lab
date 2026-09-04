#!/usr/bin/env bash
set -euo pipefail

.venv/bin/python -m megatron_lab.topology \
  --nodes "${NODES:-2}" \
  --gpus-per-node "${GPUS_PER_NODE:-8}" \
  --tp "${TP:-2}" \
  --pp "${PP:-2}" \
  --cp "${CP:-1}" \
  --ep "${EP:-1}" \
  "$@"
