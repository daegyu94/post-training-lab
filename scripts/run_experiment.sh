#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
case "$CUDA_VISIBLE_DEVICES" in
  *,*)
    echo "Expose exactly one GPU, not: $CUDA_VISIBLE_DEVICES" >&2
    exit 2
    ;;
esac

model_dir="${MODEL_DIR:-models/Qwen2.5-7B-Instruct}"
data_dir="${DATA_DIR:-data/ultrachat_200k}"
output_dir="${OUTPUT_DIR:-results/qwen2.5-7b-megatron-experiment}"
train_data="$data_dir/training.jsonl"
eval_data="$data_dir/validation.jsonl"

.venv/bin/python -m megatron_lab.preflight \
  --model-dir "$model_dir" \
  --train-data "$train_data" \
  --eval-data "$eval_data"

mkdir -p "$output_dir"
common_args=(
  --model-dir "$model_dir"
  --train-data "$train_data"
  --eval-data "$eval_data"
  --output-dir "$output_dir"
  --max-steps "${MAX_STEPS:-5}"
  --eval-iters "${EVAL_ITERS:-8}"
  --max-length "${MAX_LENGTH:-512}"
  --global-batch-size "${GLOBAL_BATCH_SIZE:-8}"
  --seed "${SEED:-42}"
)

echo "[workflow] Stage 1/3: evaluating the pretrained checkpoint"
.venv/bin/torchrun --standalone --nproc-per-node=1 \
  -m megatron_lab.sft --stage base "${common_args[@]}" "$@" \
  2>&1 | tee "$output_dir/base-eval.log"

echo "[workflow] Stage 2/3: training and saving the LoRA checkpoint"
.venv/bin/torchrun --standalone --nproc-per-node=1 \
  -m megatron_lab.sft --stage train "${common_args[@]}" "$@" \
  2>&1 | tee "$output_dir/train.log"

echo "[workflow] Stage 3/3: reloading and evaluating the LoRA checkpoint"
.venv/bin/torchrun --standalone --nproc-per-node=1 \
  -m megatron_lab.sft --stage tuned "${common_args[@]}" "$@" \
  2>&1 | tee "$output_dir/tuned-eval.log"

.venv/bin/python -m megatron_lab.compare \
  --base-log "$output_dir/base-eval.log" \
  --tuned-log "$output_dir/tuned-eval.log" \
  --output "$output_dir/summary.json"
