#!/usr/bin/env bash
set -euo pipefail

model_dir="${MODEL_DIR:-models/Qwen2.5-7B-Instruct}"

echo "Downloading Qwen/Qwen2.5-7B-Instruct to $model_dir"
echo "The BF16 checkpoint requires about 16GB of disk space."
.venv/bin/hf download Qwen/Qwen2.5-7B-Instruct --local-dir "$model_dir"
