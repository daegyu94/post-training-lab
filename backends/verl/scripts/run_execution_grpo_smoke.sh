#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname -s)" != "spark2" ]]; then
  echo "run this trainer command on spark2" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "$0")/../../.." && pwd)"
python_bin="${VERL_PYTHON:-/home/spark/.local/ptl/venvs/verl/bin/python}"
model_dir="${MODEL_DIR:-/home/spark/.local/ptl/models/Qwen2.5-0.5B-Instruct}"
adapter_dir="${LORA_ADAPTER:?set LORA_ADAPTER to the filtered-SFT adapter directory}"
tasks_path="${TASKS:?set TASKS to execution_feedback tasks.jsonl}"
data_dir="${DATA_DIR:-$repo_root/output/execution-rlvr/data}"
output_dir="${OUTPUT_DIR:-/mnt/post-training/verl/checkpoints/execution-rlvr-smoke}"
max_steps="${MAX_STEPS:-1}"
image="${EXECUTION_REWARD_IMAGE:-python:3.12-slim}"
timeout_seconds="${EXECUTION_REWARD_TIMEOUT_SECONDS:-10}"
memory="${EXECUTION_REWARD_MEMORY:-256m}"
cpus="${EXECUTION_REWARD_CPUS:-1}"
pids_limit="${EXECUTION_REWARD_PIDS_LIMIT:-64}"

[[ -x "$python_bin" ]] || { echo "missing Verl Python: $python_bin" >&2; exit 2; }
[[ -f "$model_dir/config.json" ]] || { echo "missing base model: $model_dir" >&2; exit 2; }
[[ -f "$adapter_dir/adapter_config.json" ]] || { echo "missing adapter config: $adapter_dir" >&2; exit 2; }
[[ -f "$adapter_dir/adapter_model.safetensors" ]] || { echo "missing adapter weights: $adapter_dir" >&2; exit 2; }
[[ -f "$tasks_path" ]] || { echo "missing execution tasks: $tasks_path" >&2; exit 2; }
docker image inspect "$image" >/dev/null || { echo "missing Docker image: $image" >&2; exit 2; }
mkdir -p "$output_dir/logs"
lora_rank="$("$python_bin" -c 'import json, sys; print(json.load(open(sys.argv[1]))["r"])' "$adapter_dir/adapter_config.json")"

export PATH="$(dirname "$python_bin"):$PATH"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp1s0f0np0}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-rocep1s0f0}"
export PYTHONPATH="$repo_root:$repo_root/backends/verl${PYTHONPATH:+:$PYTHONPATH}"
export EXECUTION_REWARD_IMAGE="$image"
export EXECUTION_REWARD_TIMEOUT_SECONDS="$timeout_seconds"
export EXECUTION_REWARD_MEMORY="$memory"
export EXECUTION_REWARD_CPUS="$cpus"
export EXECUTION_REWARD_PIDS_LIMIT="$pids_limit"

"$python_bin" -c 'import ray; ray.init(address="auto"); ray.shutdown()'
"$python_bin" -m verl_lab.prepare_execution --tasks "$tasks_path" --output-dir "$data_dir"

exec "$python_bin" -m verl.trainer.main_ppo \
  +ray_kwargs.ray_init.address=auto \
  +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME" \
  +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_IB_HCA="$NCCL_IB_HCA" \
  +ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH="$PYTHONPATH" \
  +ray_kwargs.ray_init.runtime_env.env_vars.EXECUTION_REWARD_IMAGE="$EXECUTION_REWARD_IMAGE" \
  +ray_kwargs.ray_init.runtime_env.env_vars.EXECUTION_REWARD_TIMEOUT_SECONDS="\"$EXECUTION_REWARD_TIMEOUT_SECONDS\"" \
  +ray_kwargs.ray_init.runtime_env.env_vars.EXECUTION_REWARD_MEMORY="$EXECUTION_REWARD_MEMORY" \
  +ray_kwargs.ray_init.runtime_env.env_vars.EXECUTION_REWARD_CPUS="\"$EXECUTION_REWARD_CPUS\"" \
  +ray_kwargs.ray_init.runtime_env.env_vars.EXECUTION_REWARD_PIDS_LIMIT="\"$EXECUTION_REWARD_PIDS_LIMIT\"" \
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
  reward.custom_reward_function.path="$repo_root/backends/verl/verl_lab/execution_reward.py" \
  reward.custom_reward_function.name=compute_score \
  actor_rollout_ref.hybrid_engine=True \
  actor_rollout_ref.model.path="$model_dir" \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.model.lora_rank="$lora_rank" \
  actor_rollout_ref.model.lora_adapter_path="$adapter_dir" \
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
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=1 \
  trainer.total_training_steps="$max_steps" \
  trainer.v1.trainer_mode=separate_async \
  trainer.v1.separate_async.parameter_sync_step=1 \
  trainer.v1.separate_async.num_warmup_batches=1 \
  trainer.logger='[console]' \
  trainer.val_before_train=False \
  trainer.save_freq="$max_steps" \
  trainer.test_freq=-1 \
  trainer.resume_mode=disable \
  trainer.default_local_dir="$output_dir" \
  hydra.run.dir="$output_dir/hydra" \
  "$@" > >(tee -a "$output_dir/logs/trainer.log") 2>&1
