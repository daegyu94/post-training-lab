#!/usr/bin/env bash
set -euo pipefail

: "${NODE_RANK:?set NODE_RANK=0 on spark1 or NODE_RANK=1 on spark2}"
: "${MASTER_ADDR:?set MASTER_ADDR to the spark1 data address}"
nnodes="${NNODES:-2}"
nproc_per_node="${NPROC_PER_NODE:-1}"
master_port="${MASTER_PORT:-29500}"
python_bin="${PYTHON:-.venv/bin/python}"
export PATH="$(dirname "$python_bin"):$PATH"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.local/ptl/cache}"
export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$XDG_CACHE_HOME/torch_extensions}"
repo_root="$(cd "$(dirname "$0")/../../.." && pwd)"
export PYTHONPATH="$repo_root/third_party/post-training-telemetry${PYTHONPATH:+:$PYTHONPATH}"
[[ -f "$repo_root/third_party/post-training-telemetry/post_training_telemetry/__init__.py" ]] || { echo "third_party/post-training-telemetry is empty; run git submodule update --init on the controller" >&2; exit 1; }
model_id="${MODEL_ID:-Qwen/Qwen3-30B-A3B}"
model_revision="${MODEL_REVISION:?set MODEL_REVISION to a 40-hex snapshot revision}"
model_dir="${MODEL_DIR:-}"
if [[ -z "$model_dir" ]]; then
  model_dir="$($python_bin -m trl_lab.spark_config --resolve-model-id "$model_id" --revision "$model_revision")"
fi
: "${DATASET_ID:?set DATASET_ID to the prepared dataset ID}"
dataset_id="$DATASET_ID"
dataset_revision="${DATASET_REVISION:?set DATASET_REVISION to a 40-hex manifest revision}"
data_dir="${DATA_DIR:?set DATA_DIR to a prepared canonical JSONL directory}"
distributed_backend="${DISTRIBUTED_BACKEND:-ddp}"
if [[ "${FSDP2:-false}" == true ]]; then
  if [[ -n "${DISTRIBUTED_BACKEND:-}" && "$distributed_backend" != fsdp2 ]]; then
    echo "FSDP2=true conflicts with DISTRIBUTED_BACKEND=$distributed_backend" >&2
    exit 2
  fi
  distributed_backend=fsdp2
fi
case "$distributed_backend" in
  ddp|fsdp2|deepspeed) ;;
  *) echo "DISTRIBUTED_BACKEND must be ddp, fsdp2, or deepspeed" >&2; exit 2 ;;
esac
deepspeed_config="${DEEPSPEED_CONFIG:-}"
if [[ "$distributed_backend" == deepspeed ]]; then
  : "${DEEPSPEED_CONFIG:?set DEEPSPEED_CONFIG to a ZeRO-2 or ZeRO-3 JSON file}"
  if [[ ! -f "$deepspeed_config" ]]; then
    echo "DEEPSPEED_CONFIG does not exist: $deepspeed_config" >&2
    exit 2
  fi
elif [[ -n "$deepspeed_config" ]]; then
  echo "DEEPSPEED_CONFIG is valid only with DISTRIBUTED_BACKEND=deepspeed" >&2
  exit 2
fi
output_dir="${OUTPUT_DIR:-results/spark-${model_id##*/}-${distributed_backend}}"
pad_to_max_length="${PAD_TO_MAX_LENGTH:-false}"
case "$pad_to_max_length" in true|false) ;; *) echo "PAD_TO_MAX_LENGTH must be true or false" >&2; exit 2 ;; esac
stage="${STAGE:-all}"
case "$stage" in all|base|train|tuned) ;; *) echo "STAGE must be all, base, train, or tuned" >&2; exit 2 ;; esac
if [[ "$distributed_backend" == fsdp2 && ( "$stage" == all || "$stage" == tuned ) ]]; then
  echo "FSDP2 currently supports STAGE=base or STAGE=train; sharded export/reload is not yet verified" >&2
  exit 2
fi
mkdir -p "$output_dir/logs"
sampler_pid=""
stop_sampler() {
  if [[ -n "$sampler_pid" ]]; then
    kill "$sampler_pid" 2>/dev/null || true
    wait "$sampler_pid" 2>/dev/null || true
  fi
}
if [[ "${RESOURCE_SAMPLING:-false}" == "true" ]]; then
  mkdir -p "$output_dir/measurements"
  "$python_bin" -m post_training_telemetry.resource_sampler \
    --target "$output_dir" \
    --output "$output_dir/measurements/resources-node-${NODE_RANK}.jsonl" &
  sampler_pid=$!
  trap stop_sampler EXIT
fi

common_args=(
  --model-id "$model_id" --model-dir "$model_dir" --model-revision "$model_revision"
  --dataset-id "$dataset_id" --dataset-revision "$dataset_revision" --data-dir "$data_dir"
  --output-dir "$output_dir"
  --finetuning-mode "${FINETUNING_MODE:-lora}" --optimizer "${OPTIMIZER:-adamw}"
  --learning-rate "${LEARNING_RATE:-2e-5}"
  --max-length "${MAX_LENGTH:-512}" --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-8}"
  --distributed-backend "$distributed_backend"
  --seed "${SEED:-42}"
  --lora-r "${LORA_R:-8}" --lora-alpha "${LORA_ALPHA:-16}"
)
if [[ "$pad_to_max_length" == "true" ]]; then common_args+=(--pad-to-max-length); fi
if [[ -n "${NUM_TRAIN_EPOCHS:-}" ]]; then
  common_args+=(--epochs "$NUM_TRAIN_EPOCHS")
else
  common_args+=(--max-steps "${MAX_STEPS:-5}")
fi
if [[ -n "${TRAIN_SAMPLES:-}" ]]; then common_args+=(--train-samples "$TRAIN_SAMPLES"); fi
if [[ -n "${EVAL_SAMPLES:-}" ]]; then common_args+=(--eval-samples "$EVAL_SAMPLES"); fi
if [[ "$distributed_backend" == deepspeed ]]; then
  common_args+=(--deepspeed-config "$deepspeed_config")
fi
if [[ -n "${LOAD_DIR:-}" ]]; then common_args+=(--load-dir "$LOAD_DIR"); fi
if [[ "${RESTORE_ONLY:-false}" == true ]]; then common_args+=(--restore-only); fi

echo "[workflow] stage=$stage rank=$NODE_RANK/$nnodes model=$model_id model_dir=$model_dir backend=$distributed_backend"
run_stage() {
  local current_stage="$1"
  shift
  if [[ "${ALLOW_EXISTING_OUTPUT:-false}" != true && -e "$output_dir/summary-${current_stage}.json" ]]; then
    echo "refusing to reuse completed stage output: $output_dir" >&2
    exit 2
  fi
  echo "[workflow] running stage=$current_stage rank=$NODE_RANK/$nnodes"
  RANK_LOG_DIR="$output_dir/logs" TELEMETRY_METRICS_DIR="${TELEMETRY_METRICS_DIR:-$output_dir/telemetry-metrics}" \
    "$python_bin" -m torch.distributed.run \
      --nnodes "$nnodes" --nproc-per-node "$nproc_per_node" --node-rank "$NODE_RANK" \
      --master-addr "$MASTER_ADDR" --master-port "$master_port" \
      -m trl_lab.spark_train "${common_args[@]}" --stage "$current_stage" "$@"
}

if [[ "$stage" == all ]]; then
  run_stage base
  run_stage train
  run_stage tuned
else
  run_stage "$stage"
fi
