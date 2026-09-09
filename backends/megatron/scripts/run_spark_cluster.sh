#!/usr/bin/env bash
set -euo pipefail

# Run this script once on each Spark node.  NODE_RANK is the only value that
# differs between spark1 and spark2.  This intentionally uses an explicit
# rendezvous; --standalone cannot coordinate two nodes.
: "${NODE_RANK:?set NODE_RANK=0 on spark1 or NODE_RANK=1 on spark2}"

nnodes="${NNODES:-2}"
nproc_per_node="${NPROC_PER_NODE:-1}"
master_addr="${MASTER_ADDR:?set MASTER_ADDR to the spark1 data IP or hostname}"
master_port="${MASTER_PORT:-29500}"
stage="${STAGE:-all}"
case "$stage" in
  all|base|train|tuned) ;;
  *) echo "STAGE must be all, base, train, or tuned" >&2; exit 2 ;;
esac
model_id="${MODEL_ID:-Qwen/Qwen3-30B-A3B}"
data_dir="${DATA_DIR:-data/public-smoke/no_robots}"
output_dir="${OUTPUT_DIR:-results/setup2-${model_id##*/}}"
python_bin="${PYTHON:-.venv/bin/python}"
export PATH="$(dirname "$python_bin"):$PATH"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.local/ptl/cache}"
export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$XDG_CACHE_HOME/torch_extensions}"
repo_root="$(cd "$(dirname "$0")/../../.." && pwd)"
export PYTHONPATH="$repo_root/observability${PYTHONPATH:+:$PYTHONPATH}"
load_checkpoint="${LOAD_CHECKPOINT:-}"
export NNODES="$nnodes" NPROC_PER_NODE="$nproc_per_node"

if [[ "${ALLOW_EXISTING_OUTPUT:-false}" != "true" && ( -e "$output_dir/summary.json" || -e "$output_dir/run-metadata-base.json" ) ]]; then
  echo "refusing to reuse completed output directory: $output_dir" >&2
  exit 2
fi

case "$model_id" in
  Qwen/Qwen3-30B-A3B)
    model_revision="${MODEL_REVISION:-ad44e777bcd18fa416d9da3bd8f70d33ebb85d39}"
    ;;
  zai-org/GLM-4.7-Flash)
    model_revision="${MODEL_REVISION:-7dd20894a642a0aa287e9827cb1a1f7f91386b67}"
    ;;
  *)
    model_revision="${MODEL_REVISION:?set MODEL_REVISION for a non-PoC feature model}"
    ;;
esac
: "${DATASET_ID:?set DATASET_ID to the prepared dataset ID}"
: "${DATASET_REVISION:?set DATASET_REVISION to a 40-hex manifest revision}"
dataset_id="$DATASET_ID"
dataset_revision="$DATASET_REVISION"

model_dir_args=()
if [[ -n "${MODEL_DIR:-}" ]]; then
  model_dir_args=(--model-dir "$MODEL_DIR")
fi
model_dir="$($python_bin -m megatron_lab.model_cache \
  --model-id "$model_id" --revision "$model_revision" \
  "${model_dir_args[@]}")"

tp="${TP:-1}"
pp="${PP:-1}"
ep="${EP:-2}"
micro_batch_size="${MICRO_BATCH_SIZE:-1}"
global_batch_size="${GLOBAL_BATCH_SIZE:-8}"
max_steps="${MAX_STEPS:-5}"
save_interval="${SAVE_INTERVAL:-}"
if [[ -n "$save_interval" && ! "$save_interval" =~ ^[1-9][0-9]*$ ]]; then
  echo "SAVE_INTERVAL must be a positive integer" >&2
  exit 2
fi
if [[ -n "$load_checkpoint" && ! -d "$load_checkpoint" ]]; then
  echo "LOAD_CHECKPOINT must name an existing checkpoint directory" >&2
  exit 2
fi
schedule_steps="${SCHEDULE_STEPS:-${RESUME_MAX_STEPS:-$max_steps}}"
eval_iters="${EVAL_ITERS:-8}"
max_length="${MAX_LENGTH:-2048}"
pad_to_max_length="${PAD_TO_MAX_LENGTH:-false}"
case "$pad_to_max_length" in true|false) ;; *) echo "PAD_TO_MAX_LENGTH must be true or false" >&2; exit 2 ;; esac
seed="${SEED:-42}"
transformer_impl="${TRANSFORMER_IMPL:-auto}"
dist_ckpt_optim_fully_reshardable="${DIST_CKPT_OPTIM_FULLY_RESHARDABLE:-false}"

case "$dist_ckpt_optim_fully_reshardable" in
  true|false) ;;
  *)
    echo "DIST_CKPT_OPTIM_FULLY_RESHARDABLE must be true or false" >&2
    exit 2
    ;;
esac
if [[ "$dist_ckpt_optim_fully_reshardable" == "true" && "${DISTRIBUTED_OPTIMIZER:-true}" == "false" ]]; then
  echo "fully reshardable optimizer checkpoints require DISTRIBUTED_OPTIMIZER=true" >&2
  exit 2
fi
if [[ "$dist_ckpt_optim_fully_reshardable" == "true" && "${SAVE_OPTIMIZER:-true}" == "false" ]]; then
  echo "fully reshardable checkpoint creation requires SAVE_OPTIMIZER=true" >&2
  exit 2
fi

"$python_bin" -m megatron_lab.cluster \
  --nnodes "$nnodes" \
  --node-rank "$NODE_RANK" \
  --nproc-per-node "$nproc_per_node" \
  --tp "$tp" --pp "$pp" --ep "$ep" \
  --micro-batch-size "$micro_batch_size" \
  --global-batch-size "$global_batch_size"

mkdir -p "$output_dir/logs"
common_args=(
  --setup spark-cluster
  --model-id "$model_id"
  --model-revision "$model_revision"
  --dataset-revision "$dataset_revision"
  --dataset-id "$dataset_id"
  --model-dir "$model_dir"
  --train-data "$data_dir/training.jsonl"
  --eval-data "$data_dir/validation.jsonl"
  --output-dir "$output_dir"
  --max-steps "$max_steps"
  --schedule-steps "$schedule_steps"
  --eval-iters "$eval_iters"
  --max-length "$max_length"
  --global-batch-size "$global_batch_size"
  --micro-batch-size "$micro_batch_size"
  --tp "$tp" --pp "$pp" --ep "$ep"
  --seed "$seed"
  --transformer-impl "$transformer_impl"
  --finetuning-mode "${FINETUNING_MODE:-lora}"
  --checkpoint-mode "${CHECKPOINT_MODE:-sync}"
  --recompute "${RECOMPUTE:-full}"
)
if [[ "$dist_ckpt_optim_fully_reshardable" == "true" ]]; then
  common_args+=(--dist-ckpt-optim-fully-reshardable)
fi
if [[ -n "$save_interval" ]]; then
  common_args+=(--save-interval "$save_interval")
fi
if [[ "$pad_to_max_length" == "true" ]]; then
  common_args+=(--pad-to-max-length)
fi

if [[ "${DISTRIBUTED_OPTIMIZER:-true}" == "false" ]]; then
  common_args+=(--no-distributed-optimizer)
fi
if [[ "${SAVE_OPTIMIZER:-true}" == "false" ]]; then
  common_args+=(--no-save-optimizer)
fi
if [[ "${LOAD_OPTIMIZER:-true}" == "false" ]]; then
  common_args+=(--no-load-optimizer)
fi
if [[ "${RESUME_AFTER_TRAIN:-false}" == "true" ]]; then
  resume_steps="${RESUME_MAX_STEPS:?set RESUME_MAX_STEPS greater than MAX_STEPS for true resume}"
  if (( resume_steps <= max_steps )); then
    echo "RESUME_MAX_STEPS must be greater than MAX_STEPS" >&2
    exit 2
  fi
  resume_tp="${RESUME_TP:-$tp}"
  resume_pp="${RESUME_PP:-$pp}"
  resume_ep="${RESUME_EP:-$ep}"
fi
script_args=("$@")

if [[ "${SEQUENCE_PARALLEL:-false}" == "true" ]]; then
  common_args+=(--sequence-parallel)
fi
if [[ "${OVERLAP_GRAD_REDUCE:-false}" == "true" ]]; then
  common_args+=(--overlap-grad-reduce)
fi
if [[ "${FULLY_PARALLEL_SAVE:-true}" == "false" ]]; then
  common_args+=(--no-fully-parallel-save)
fi
if [[ "${FULLY_PARALLEL_LOAD:-true}" == "false" ]]; then
  common_args+=(--no-fully-parallel-load)
fi

run_stage() {
  local stage="$1"
  local -a stage_args=("${common_args[@]}" "${script_args[@]}")
  if [[ "$stage" == "resume" && -n "$load_checkpoint" ]]; then
    stage_args+=(--load-checkpoint "$load_checkpoint")
  elif [[ "$stage" == "tuned" && -n "$load_checkpoint" && "${RESUME_AFTER_TRAIN:-false}" != "true" ]]; then
    stage_args+=(--load-checkpoint "$load_checkpoint")
  fi
  if [[ "$stage" == "resume" || ( "$stage" == "tuned" && "${RESUME_AFTER_TRAIN:-false}" == "true" ) ]]; then
    stage_args+=(
      --max-steps "$resume_steps"
      --tp "$resume_tp" --pp "$resume_pp" --ep "$resume_ep"
    )
  fi
  echo "[workflow] stage=$stage node_rank=$NODE_RANK"
  RANK_LOG_DIR="$output_dir/logs" FRAMEWORK_METRICS_DIR="$output_dir/framework-metrics" \
    "$python_bin" -m torch.distributed.run \
      --nnodes "$nnodes" \
      --nproc-per-node "$nproc_per_node" \
      --node-rank "$NODE_RANK" \
      --master-addr "$master_addr" \
      --master-port "$master_port" \
      -m megatron_lab.sft --stage "$stage" "${stage_args[@]}"
}

if [[ "$stage" != all ]]; then
  run_stage "$stage"
  exit 0
fi

run_stage base
run_stage train
if [[ "${RESUME_AFTER_TRAIN:-false}" == "true" ]]; then
  run_stage resume
fi
run_stage tuned

# Bridge emits validation loss on the last global rank.
metrics_rank=$((nnodes * nproc_per_node - 1))
if (( NODE_RANK == nnodes - 1 )); then
  compare_max_steps="$max_steps"
  compare_tp="$tp"
  compare_pp="$pp"
  compare_ep="$ep"
  if [[ "${RESUME_AFTER_TRAIN:-false}" == "true" ]]; then
    compare_max_steps="$resume_steps"
    compare_tp="$resume_tp"
    compare_pp="$resume_pp"
    compare_ep="$resume_ep"
  fi
  "$python_bin" -m megatron_lab.compare \
    --base-log "$output_dir/logs/rank-${metrics_rank}-base.log" \
    --tuned-log "$output_dir/logs/rank-${metrics_rank}-tuned.log" \
    --output "$output_dir/summary.json" \
    --model-id "$model_id" \
    --model-revision "$model_revision" \
    --dataset-revision "$dataset_revision" \
    --dataset-id "$dataset_id" \
    --model-dir "$model_dir" \
    --train-data "$data_dir/training.jsonl" \
    --eval-data "$data_dir/validation.jsonl" \
    --output-dir "$output_dir" \
    --max-steps "$compare_max_steps" \
    --schedule-steps "$schedule_steps" \
    --eval-iters "$eval_iters" \
    --max-length "$max_length" \
    --global-batch-size "$global_batch_size" \
    --micro-batch-size "$micro_batch_size" \
    --seed "$seed" \
    --finetuning-mode "${FINETUNING_MODE:-lora}" \
    --metadata "$output_dir/run-metadata-tuned-rank-${metrics_rank}.json" \
    --tp "$compare_tp" --pp "$compare_pp" --ep "$compare_ep" \
    --nnodes "$nnodes" --nproc-per-node "$nproc_per_node"
fi
