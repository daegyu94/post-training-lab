#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
run_id="${RUN_ID:?set RUN_ID to a new identifier}"
if [[ ! "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "RUN_ID may contain only letters, numbers, dots, underscores, and hyphens" >&2
  exit 2
fi

ssh_config="${SSH_CONFIG:-/dev/null}"
spark1_host="${SPARK1_HOST:-spark@spark1}"
spark_repo="${SPARK_REPO:-/home/spark/shared/$(basename "$repo_root")}"
spark1_work_dir="${SPARK1_WORK_ROOT:-/mnt/post-training/execution-rlvr}/$run_id"
stage_dir="${STAGE_ROOT:-/home/spark/shared/execution-rlvr}/$run_id"
model_dir="${MODEL_DIR:-/home/spark/.local/ptl/models/Qwen2.5-0.5B-Instruct}"

run_remote() {
  local host="$1"
  shift
  ssh -F "$ssh_config" -o BatchMode=yes -o ConnectTimeout=10 "$host" "$@"
}

run_remote "$spark1_host" env \
  TRL_PYTHON="${TRL_PYTHON:-/home/spark/.local/ptl/venvs/trl/bin/python}" \
  MODEL_DIR="$model_dir" WORK_DIR="$spark1_work_dir" STAGE_DIR="$stage_dir" \
  LIMIT_PER_SPLIT="${LIMIT_PER_SPLIT:-2}" INITIAL_SFT_STEPS="${INITIAL_SFT_STEPS:-32}" \
  FILTERED_SFT_STEPS="${FILTERED_SFT_STEPS:-1}" LORA_RANK="${LORA_RANK:-8}" \
  RLVR_STEPS="${RLVR_MAX_STEPS:-1}" \
  NUM_TRAIN_CANDIDATES="${NUM_TRAIN_CANDIDATES:-8}" MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}" \
  DOCKER_WORKERS="${DOCKER_WORKERS:-1}" \
  bash "$spark_repo/scripts/run_execution_filtered_sft_smoke.sh"
