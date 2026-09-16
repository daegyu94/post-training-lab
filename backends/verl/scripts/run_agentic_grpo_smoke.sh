#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname -s)" != "spark2" ]]; then
  echo "run this trainer command on spark2" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "$0")/../../.." && pwd)"
python_bin="${VERL_PYTHON:-/home/spark/.local/ptl/venvs/verl/bin/python}"
model_dir="${MODEL_DIR:-/home/spark/.local/ptl/models/Qwen2.5-0.5B-Instruct}"
lora_rank="${LORA_RANK:-8}"
data_dir="${DATA_DIR:-$repo_root/data/verl/calculator-smoke}"
output_dir="${OUTPUT_DIR:-/mnt/post-training/verl/checkpoints/calculator-smoke}"

[[ -x "$python_bin" ]] || { echo "missing Verl Python: $python_bin" >&2; exit 2; }
[[ -f "$model_dir/config.json" ]] || { echo "missing model: $model_dir" >&2; exit 2; }
mkdir -p "$output_dir/logs"
export PATH="$(dirname "$python_bin"):$PATH"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp1s0f0np0}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-rocep1s0f0}"
"$python_bin" -c 'import ray; ray.init(address="auto"); ray.shutdown()'

export PYTHONPATH="$repo_root/backends/verl${PYTHONPATH:+:$PYTHONPATH}"
"$python_bin" -m verl_lab.prepare_smoke "$data_dir"

exec "$python_bin" -m verl.trainer.main_ppo \
  +ray_kwargs.ray_init.address=auto \
  +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME" \
  +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_IB_HCA="$NCCL_IB_HCA" \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="$data_dir/train.parquet" \
  data.val_files="$data_dir/val.parquet" \
  data.train_batch_size=2 \
  data.gen_batch_size=2 \
  data.max_prompt_length=256 \
  data.max_response_length=128 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  reward.custom_reward_function.path="$repo_root/backends/verl/verl_lab/reward.py" \
  reward.custom_reward_function.name=compute_score \
  actor_rollout_ref.hybrid_engine=True \
  actor_rollout_ref.model.path="$model_dir" \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.model.lora_rank="$lora_rank" \
  actor_rollout_ref.model.lora.merge=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.nnodes=1 \
  actor_rollout_ref.rollout.n_gpus_per_node=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.max_num_batched_tokens=512 \
  actor_rollout_ref.rollout.max_num_seqs=4 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
  +actor_rollout_ref.rollout.checkpoint_engine.custom_backend_module=verl.checkpoint_engine.nccl_checkpoint_engine \
  actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.format=hermes \
  actor_rollout_ref.rollout.multi_turn.function_tool_path="$repo_root/backends/verl/verl_lab/function_tools.py" \
  actor_rollout_ref.rollout.multi_turn.max_user_turns=1 \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=2 \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=1 \
  trainer.total_training_steps=1 \
  trainer.v1.trainer_mode=separate_async \
  trainer.v1.separate_async.parameter_sync_step=1 \
  trainer.v1.separate_async.num_warmup_batches=1 \
  trainer.logger='[console]' \
  trainer.val_before_train=False \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.resume_mode=disable \
  trainer.default_local_dir="$output_dir" \
  hydra.run.dir="$output_dir/hydra" \
  "$@" > >(tee -a "$output_dir/logs/trainer.log") 2>&1
