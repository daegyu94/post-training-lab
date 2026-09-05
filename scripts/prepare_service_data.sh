#!/usr/bin/env bash
set -euo pipefail

.venv/bin/python -m sft_lab.prepare_service_data \
  --input "${TRACE_FILE:-examples/service-traces.jsonl}" \
  --output-dir "${DATA_DIR:-data/service-sft}" \
  --seed "${SEED:-42}" \
  --train-ratio "${TRAIN_RATIO:-0.8}" \
  --validation-ratio "${VALIDATION_RATIO:-0.1}"
