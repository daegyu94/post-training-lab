#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
WORK_DIR="${WORK_DIR:-$ROOT_DIR/output/execution-feedback-cycle}"
SFT_CHECKPOINT="${SFT_CHECKPOINT:?set SFT_CHECKPOINT to the common starting checkpoint}"
TRAIN_LAUNCH="${TRAIN_LAUNCH:-$PYTHON}"
DATA_SOURCE="${DATA_SOURCE:-synthetic}"
NUM_TRAIN_CANDIDATES="${NUM_TRAIN_CANDIDATES:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TRAIN_TEMPERATURE="${TRAIN_TEMPERATURE:-0.8}"
TRAIN_TOP_P="${TRAIN_TOP_P:-0.95}"
DOCKER_WORKERS="${DOCKER_WORKERS:-1}"
LORA_R="${LORA_R:-0}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"
MAX_STEPS="${MAX_STEPS:-64}"

read -r -a train_launch <<<"$TRAIN_LAUNCH"
mkdir -p "$WORK_DIR"
cd "$ROOT_DIR"

prepare_args=(--source "$DATA_SOURCE" --output-dir "$WORK_DIR/data")
if [[ "$DATA_SOURCE" == "mbpp" ]]; then
  : "${DATA_REVISION:?set DATA_REVISION to an immutable 40-character MBPP revision}"
  prepare_args+=(--revision "$DATA_REVISION")
fi
"$PYTHON" -m execution_feedback.prepare "${prepare_args[@]}"

"$PYTHON" -m execution_feedback.generate --model-dir "$SFT_CHECKPOINT" \
  --tasks "$WORK_DIR/data/tasks.jsonl" --split train --num-candidates "$NUM_TRAIN_CANDIDATES" \
  --seed 42 --temperature "$TRAIN_TEMPERATURE" --top-p "$TRAIN_TOP_P" --max-new-tokens "$MAX_NEW_TOKENS" --output "$WORK_DIR/train-candidates.jsonl"
"$PYTHON" -m execution_feedback.evaluate --tasks "$WORK_DIR/data/tasks.jsonl" \
  --candidates "$WORK_DIR/train-candidates.jsonl" --output "$WORK_DIR/train-evaluations.jsonl" \
  --engine docker --workers "$DOCKER_WORKERS"
"$PYTHON" -m execution_feedback.feedback --tasks "$WORK_DIR/data/tasks.jsonl" \
  --evaluations "$WORK_DIR/train-evaluations.jsonl" --output-dir "$WORK_DIR/feedback"

if [[ ! -s "$WORK_DIR/feedback/filtered_sft.jsonl" ]]; then
  echo "no passing train candidates; cannot run additional training" >&2
  exit 2
fi

common_train=(--max-steps "$MAX_STEPS" --per-device-batch-size 1 --gradient-accumulation-steps 4 --seed 42)
if [[ "$GRADIENT_CHECKPOINTING" == 1 ]]; then
  common_train+=(--gradient-checkpointing)
fi
"${train_launch[@]}" -m execution_feedback.train --mode sft --model-dir "$SFT_CHECKPOINT" \
  --train-file "$WORK_DIR/feedback/filtered_sft.jsonl" --eval-file "$WORK_DIR/data/initial_sft_validation.jsonl" \
  --output-dir "$WORK_DIR/checkpoints/B" "${common_train[@]}" --lora-r "$LORA_R"

declare -A models=( [A]="$SFT_CHECKPOINT" [B]="$WORK_DIR/checkpoints/B/model" )
if [[ -s "$WORK_DIR/feedback/dpo.jsonl" ]]; then
  "${train_launch[@]}" -m execution_feedback.train --mode dpo --model-dir "$SFT_CHECKPOINT" \
    --train-file "$WORK_DIR/feedback/dpo.jsonl" --output-dir "$WORK_DIR/checkpoints/C" "${common_train[@]}" --lora-r "$LORA_R"
  "${train_launch[@]}" -m execution_feedback.train --mode dpo --model-dir "$WORK_DIR/checkpoints/B/model" \
    --train-file "$WORK_DIR/feedback/dpo.jsonl" --output-dir "$WORK_DIR/checkpoints/D" "${common_train[@]}"
  models[C]="$WORK_DIR/checkpoints/C/model"
  models[D]="$WORK_DIR/checkpoints/D/model"
else
  echo "no valid pass/fail pairs; skipping C and D" >&2
fi

compare_args=()
for variant in A B C D; do
  [[ -n "${models[$variant]:-}" ]] || continue
  "$PYTHON" -m execution_feedback.generate --model-dir "${models[$variant]}" \
    --tasks "$WORK_DIR/data/tasks.jsonl" --split test --num-candidates 1 \
    --seed 42 --temperature 0 --max-new-tokens "$MAX_NEW_TOKENS" --output "$WORK_DIR/test-candidates-$variant.jsonl"
  "$PYTHON" -m execution_feedback.evaluate --tasks "$WORK_DIR/data/tasks.jsonl" \
    --candidates "$WORK_DIR/test-candidates-$variant.jsonl" --output "$WORK_DIR/test-evaluations-$variant.jsonl" \
    --engine docker --workers "$DOCKER_WORKERS"
  compare_args+=(--variant "$variant=$WORK_DIR/test-evaluations-$variant.jsonl")
  if [[ "$variant" == "D" ]]; then
    compare_args+=(--training-summary "D=$WORK_DIR/checkpoints/B/execution_feedback_training.json")
  fi
  if [[ "$variant" != "A" ]]; then
    compare_args+=(--training-summary "$variant=$WORK_DIR/checkpoints/$variant/execution_feedback_training.json")
  fi
done
"$PYTHON" -m execution_feedback.compare "${compare_args[@]}" --output "$WORK_DIR/comparison.json"

echo "cycle artifacts: $WORK_DIR"
