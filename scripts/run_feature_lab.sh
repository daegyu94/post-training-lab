#!/usr/bin/env bash
set -euo pipefail

: "${NODE_RANK:?set NODE_RANK=0 on spark1 or NODE_RANK=1 on spark2}"
: "${MASTER_ADDR:?set MASTER_ADDR to the spark1 data IP or hostname}"

feature="${FEATURE:?set FEATURE (checkpoint, overlap-grad-reduce, recompute, sequence-parallel, expert-parallel)}"
case "$feature" in
  checkpoint) variants=(sync async) ;;
  overlap-grad-reduce) variants=(off on) ;;
  recompute) variants=(full selective) ;;
  sequence-parallel) variants=(off on) ;;
  expert-parallel) variants=(ep1 ep2) ;;
  *) echo "unknown FEATURE: $feature" >&2; exit 2 ;;
esac

repeats="${REPEATS:-3}"
warmup="${WARMUP:-1}"
python_bin="${PYTHON:-.venv/bin/python}"
root_output="${OUTPUT_DIR:-results/features/$feature}"
git_commit="$(git rev-parse HEAD)"
feature_model_id="${FEATURE_MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}"
feature_model_revision="${FEATURE_MODEL_REVISION:-7ae557604adf67be50417f59c2c2f167def9a775}"
feature_model_dir="${FEATURE_MODEL_DIR:-models/Qwen2.5-0.5B-Instruct}"
feature_finetuning_mode=full
if [[ "$feature" == "expert-parallel" && -z "${FEATURE_MODEL_ID:-}" ]]; then
  feature_model_id=Qwen/Qwen3-30B-A3B
  feature_model_dir="${FEATURE_MODEL_DIR:-}"
  feature_model_revision=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39
  feature_finetuning_mode=lora
fi
dataset_revision="${DATASET_REVISION:-e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b}"

total_runs=$((warmup + repeats * 2))
for run_index in $(seq 0 "$((total_runs - 1))"); do
  if (( run_index < warmup )); then
    is_warmup=true
    steps=1
  else
    is_warmup=false
    steps="${MAX_STEPS:-5}"
  fi
  variant_index=0
  if (( run_index >= warmup )); then
    variant_index=$(((run_index - warmup) % 2))
  fi
  variant="${variants[$variant_index]}"
  run_dir="$root_output/$variant/run-$run_index"
  mkdir -p "$run_dir"

  tp=1
  ep=1
  sequence_parallel=false
  overlap_grad_reduce=false
  checkpoint_mode=sync
  recompute=full
  transformer_impl=auto
  case "$feature:$variant" in
    checkpoint:async) checkpoint_mode=async ;;
    overlap-grad-reduce:on) overlap_grad_reduce=true ;;
    recompute:selective) recompute=selective ;;
    sequence-parallel:off) tp=2; transformer_impl=transformer_engine ;;
    sequence-parallel:on) tp=2; sequence_parallel=true; transformer_impl=transformer_engine ;;
    expert-parallel:ep2) ep=2 ;;
  esac

  start=$SECONDS
  NODE_RANK="$NODE_RANK" \
  MASTER_ADDR="$MASTER_ADDR" \
  MODEL_ID="$feature_model_id" \
  MODEL_DIR="$feature_model_dir" \
  MODEL_REVISION="$feature_model_revision" \
  DATASET_REVISION="$dataset_revision" \
  OUTPUT_DIR="$run_dir" \
  FINETUNING_MODE="$feature_finetuning_mode" \
  TP="$tp" EP="$ep" \
  SEQUENCE_PARALLEL="$sequence_parallel" \
  OVERLAP_GRAD_REDUCE="$overlap_grad_reduce" \
  TRANSFORMER_IMPL="$transformer_impl" \
  CHECKPOINT_MODE="$checkpoint_mode" \
  RECOMPUTE="$recompute" \
  RESUME_AFTER_TRAIN="${RESUME_AFTER_TRAIN:-false}" \
  MAX_STEPS="$steps" \
  scripts/run_spark_cluster.sh
  elapsed=$((SECONDS - start))

  if [[ "$NODE_RANK" == "0" ]]; then
    rank_log="$run_dir/logs/rank-0-train.log"
    evidence="$run_dir/rank-0-evidence.json"
    parse_warmup_steps="${STEADY_WARMUP_STEPS:-2}"
    if [[ "$is_warmup" == true ]]; then
      parse_warmup_steps=0
    fi
    "$python_bin" -m megatron_lab.feature_lab \
      --log "$rank_log" --feature "$feature" --variant "$variant" \
      --run-index "$run_index" --warmup-steps "$parse_warmup_steps" \
      --exit-code 0 --completed --output "$evidence"
    warmup_args=()
    if [[ "$is_warmup" == true ]]; then
      warmup_args+=(--warmup)
    fi
    "$python_bin" -m megatron_lab.feature_lab \
      --feature "$feature" --variant "$variant" \
      --run-index "$run_index" \
      "${warmup_args[@]}" \
      --elapsed-seconds "$elapsed" \
      --exit-code 0 \
      --model-id "$feature_model_id" \
      --model-revision "$feature_model_revision" \
      --dataset-id "${DATASET_ID:-HuggingFaceH4/no_robots}" \
      --dataset-revision "$dataset_revision" \
      --git-commit "$git_commit" \
      --metadata "$run_dir/run-metadata-tuned.json" \
      --evidence "$evidence" \
      --output "$run_dir/feature-record.json"
  fi
done

if [[ "$NODE_RANK" == "0" ]]; then
  "$python_bin" -m megatron_lab.feature_lab \
    --record-glob "$root_output/*/run-*/feature-record.json" \
    --output "$root_output/feature-summary.json"
fi
