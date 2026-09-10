#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

python_bin="${PYTHON:?Set PYTHON to a backend venv interpreter}"
"$python_bin" -m datasets_lab.prepare_service_data \
  --input "${TRACE_FILE:-datasets_lab/examples/service-traces.jsonl}" \
  --output-dir "${DATA_DIR:-data/service-sft}" \
  --seed "${SEED:-42}" \
  --train-ratio "${TRAIN_RATIO:-0.8}" \
  --validation-ratio "${VALIDATION_RATIO:-0.1}"
