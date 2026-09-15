#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname -s)" != "spark1" ]]; then
  echo "run this filtered-SFT command on spark1" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
python_bin="${TRL_PYTHON:-/home/spark/.local/ptl/venvs/trl/bin/python}"
model_dir="${MODEL_DIR:-/home/spark/.local/ptl/models/Qwen2.5-0.5B-Instruct}"
work_dir="${WORK_DIR:?set WORK_DIR to a new spark1-local output path}"
stage_dir="${STAGE_DIR:?set STAGE_DIR to a new NFS output path}"
limit_per_split="${LIMIT_PER_SPLIT:-2}"
initial_sft_steps="${INITIAL_SFT_STEPS:-32}"
filtered_sft_steps="${FILTERED_SFT_STEPS:-1}"
rlvr_steps="${RLVR_STEPS:-1}"
lora_rank="${LORA_RANK:-8}"
num_candidates="${NUM_TRAIN_CANDIDATES:-8}"
max_new_tokens="${MAX_NEW_TOKENS:-128}"
docker_workers="${DOCKER_WORKERS:-1}"

[[ -x "$python_bin" ]] || { echo "missing TRL Python: $python_bin" >&2; exit 2; }
[[ -f "$model_dir/config.json" ]] || { echo "missing base model: $model_dir" >&2; exit 2; }
[[ ! -e "$work_dir" ]] || { echo "WORK_DIR already exists: $work_dir" >&2; exit 2; }
[[ ! -e "$stage_dir" ]] || { echo "STAGE_DIR already exists: $stage_dir" >&2; exit 2; }
docker image inspect python:3.12-slim >/dev/null || { echo "missing Docker image: python:3.12-slim" >&2; exit 2; }

export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$work_dir" "$stage_dir"

"$python_bin" -m execution_feedback.prepare \
  --limit-per-split "$limit_per_split" --output-dir "$work_dir/data"
"$python_bin" -m execution_feedback.train --mode sft --model-dir "$model_dir" \
  --train-file "$work_dir/data/initial_sft_train.jsonl" \
  --eval-file "$work_dir/data/initial_sft_validation.jsonl" \
  --output-dir "$work_dir/checkpoints/initial" \
  --max-steps "$initial_sft_steps" --per-device-batch-size 1 \
  --gradient-accumulation-steps 1 --lora-r "$lora_rank"
"$python_bin" -m execution_feedback.generate \
  --model-dir "$work_dir/checkpoints/initial/model" --tasks "$work_dir/data/tasks.jsonl" \
  --split train --num-candidates "$num_candidates" --seed 42 --temperature 0.8 \
  --top-p 0.95 --max-new-tokens "$max_new_tokens" \
  --output "$work_dir/train-candidates.jsonl"
"$python_bin" -m execution_feedback.evaluate --tasks "$work_dir/data/tasks.jsonl" \
  --candidates "$work_dir/train-candidates.jsonl" \
  --output "$work_dir/train-evaluations.jsonl" --engine docker --workers "$docker_workers"
"$python_bin" -m execution_feedback.feedback --tasks "$work_dir/data/tasks.jsonl" \
  --evaluations "$work_dir/train-evaluations.jsonl" --output-dir "$work_dir/feedback"

if [[ ! -s "$work_dir/feedback/filtered_sft.jsonl" ]]; then
  echo "no passing generated code; filtered-SFT adapter was not created" >&2
  exit 2
fi

"$python_bin" -m execution_feedback.train --mode sft \
  --model-dir "$work_dir/checkpoints/initial/model" \
  --train-file "$work_dir/feedback/filtered_sft.jsonl" \
  --eval-file "$work_dir/data/initial_sft_validation.jsonl" \
  --output-dir "$work_dir/checkpoints/filtered" \
  --max-steps "$filtered_sft_steps" --per-device-batch-size 1 \
  --gradient-accumulation-steps 1
"$python_bin" -m execution_feedback.rlvr \
  --model-dir "$work_dir/checkpoints/filtered/model" --tasks "$work_dir/data/tasks.jsonl" \
  --output-dir "$work_dir/checkpoints/rlvr" --max-steps "$rlvr_steps" \
  --num-generations 2 --max-completion-length "$max_new_tokens"

cp "$work_dir/data/tasks.jsonl" "$stage_dir/tasks.jsonl"
cp "$work_dir/feedback/manifest.json" "$stage_dir/feedback-manifest.json"
cp "$work_dir/checkpoints/filtered/model/adapter_config.json" "$stage_dir/adapter_config.json"
cp "$work_dir/checkpoints/filtered/model/adapter_model.safetensors" "$stage_dir/adapter_model.safetensors"
cp "$work_dir/checkpoints/filtered/execution_feedback_training.json" "$stage_dir/filtered-sft-training.json"
mkdir "$stage_dir/rlvr-adapter"
cp "$work_dir/checkpoints/rlvr/model/adapter_config.json" "$stage_dir/rlvr-adapter/adapter_config.json"
cp "$work_dir/checkpoints/rlvr/model/adapter_model.safetensors" "$stage_dir/rlvr-adapter/adapter_model.safetensors"

echo "filtered-SFT stage artifacts: $stage_dir"
