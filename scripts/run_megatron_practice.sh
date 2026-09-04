#!/usr/bin/env bash
set -euo pipefail

echo "[practice] Inspecting the Qwen2.5-7B LoRA recipe without GPU initialization"
./scripts/inspect_megatron_recipe.sh

echo "[practice] Simulating TP=2, PP=2, CP=2, DP=2 without distributed communication"
WORLD_SIZE="${WORLD_SIZE:-16}" \
TP_SIZE="${TP_SIZE:-2}" \
PP_SIZE="${PP_SIZE:-2}" \
CP_SIZE="${CP_SIZE:-2}" \
./scripts/simulate_parallelism.sh
