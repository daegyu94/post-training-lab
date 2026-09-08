#!/usr/bin/env bash
set -euo pipefail

.venv/bin/python -m megatron_lab.parallelism \
  --world-size "${WORLD_SIZE:-8}" \
  --tensor-parallel-size "${TP_SIZE:-2}" \
  --pipeline-parallel-size "${PP_SIZE:-2}" \
  --context-parallel-size "${CP_SIZE:-1}" \
  "$@"
