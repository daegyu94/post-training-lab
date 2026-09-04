#!/usr/bin/env bash
set -euo pipefail

venv_dir="${VENV_DIR:-.venv-megatron}"
model_dir="${MODEL_DIR:-models/Qwen2.5-14B-Instruct}"

echo "Downloading Qwen/Qwen2.5-14B-Instruct to $model_dir"
echo "The BF16 checkpoint requires about 28GB of disk space."
"$venv_dir/bin/hf" download Qwen/Qwen2.5-14B-Instruct --local-dir "$model_dir"
