#!/usr/bin/env bash
set -euo pipefail

nemo_root="${NEMO_RL_ROOT:-/opt/nemo-rl}"
python_bin="${NEMO_RL_PYTHON:-python}"
model_dir="${MODEL_DIR:-/models/Qwen2.5-0.5B-Instruct}"
data_path="${DATA_PATH:-/workspace/post-training-lab/experiments/rl-framework-comparison/math-smoke.jsonl}"
output_dir="${OUTPUT_DIR:-/results/nemo-rl}"

[[ -f "$nemo_root/examples/run_grpo.py" ]] || { echo "missing NeMo RL source: $nemo_root" >&2; exit 2; }
[[ -f "$model_dir/config.json" ]] || { echo "missing model: $model_dir" >&2; exit 2; }
[[ -f "$data_path" ]] || { echo "missing data: $data_path" >&2; exit 2; }
mkdir -p "$output_dir/logs"
export NEMO_RL_PY_EXECUTABLES_SYSTEM=0
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp1s0f0np0}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-rocep1s0f0}"

exec "$python_bin" "$nemo_root/examples/run_grpo.py" \
  grpo.num_prompts_per_step=2 \
  grpo.num_generations_per_prompt=2 \
  grpo.seed=42 \
  grpo.max_num_steps=1 \
  grpo.val_period=-1 \
  grpo.val_at_start=false \
  grpo.val_at_end=false \
  policy.model_name="$model_dir" \
  policy.train_global_batch_size=4 \
  policy.train_micro_batch_size=1 \
  policy.logprob_batch_size=1 \
  policy.max_total_sequence_length=384 \
  policy.dtensor_cfg.lora_cfg.enabled=true \
  policy.dtensor_cfg.lora_cfg.dim=8 \
  policy.dtensor_cfg.lora_cfg.alpha=16 \
  policy.dtensor_cfg.lora_cfg.use_triton=false \
  policy.generation.max_new_tokens=128 \
  policy.generation.vllm_cfg.max_model_len=384 \
  policy.generation.vllm_cfg.gpu_memory_utilization=0.5 \
  policy.generation.vllm_cfg.enforce_eager=true \
  policy.generation.colocated.enabled=false \
  policy.generation.colocated.resources.gpus_per_node=1 \
  policy.generation.colocated.resources.num_nodes=1 \
  data.max_input_seq_length=256 \
  data.shuffle=false \
  data.num_workers=1 \
  data.train.dataset_name=ResponseDataset \
  +data.train.data_path="$data_path" \
  +data.train.input_key=input \
  +data.train.output_key=output \
  data.train.split_validation_size=0 \
  data.validation=null \
  data.default.prompt_file=null \
  data.default.system_prompt_file=null \
  data.default.processor=math_hf_data_processor \
  data.default.env_name=math \
  env.math.num_workers=1 \
  cluster.gpus_per_node=1 \
  cluster.num_nodes=2 \
  checkpointing.enabled=false \
  logger.log_dir="$output_dir/logs" \
  logger.wandb_enabled=false \
  logger.tensorboard_enabled=true \
  logger.monitor_gpus=true \
  "$@" 2>&1 | tee "$output_dir/trainer.log"
