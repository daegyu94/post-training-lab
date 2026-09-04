#!/usr/bin/env bash
set -euo pipefail

.venv/bin/python -m megatron_lab.topology \
  --world-size "${WORLD_SIZE:-2}" \
  "$@"
