"""Trainer/Accelerate DDP, FSDP2, or DeepSpeed SFT on Spark snapshots."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

from trl_lab.spark_config import SparkConfig, validate_config, model_snapshot_evidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("base", "train", "tuned"), required=True)
    parser.add_argument("--finetuning-mode", choices=("lora", "full"), default="lora")
    parser.add_argument("--optimizer", choices=("adamw", "sgd"), default="adamw")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--train-samples", type=int)
    parser.add_argument("--eval-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument(
        "--distributed-backend",
        choices=("ddp", "fsdp2", "deepspeed"),
        default="ddp",
    )
    parser.add_argument("--deepspeed-config", type=Path)
    return parser.parse_args()


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _destroy_process_group_if_initialized() -> None:
    """Release a torchrun process group without masking the training result."""

    torch = sys.modules.get("torch")
    distributed = getattr(torch, "distributed", None)
    if distributed is None:
        return
    try:
        if distributed.is_available() and distributed.is_initialized():
            distributed.destroy_process_group()
    except Exception as exc:  # pragma: no cover - defensive cleanup boundary
        print(f"[cleanup] destroy_process_group failed: {exc}", file=sys.__stderr__)


class _ProcessGroupCleanupContext:
    """Run process-group cleanup after either normal or exceptional exit."""

    def __init__(self, context: Any) -> None:
        self._context = context

    def __enter__(self) -> Any:
        return self._context.__enter__()

    def __exit__(self, *exc: object) -> Any:
        try:
            return self._context.__exit__(*exc)
        finally:
            _destroy_process_group_if_initialized()


def _write_log(args: argparse.Namespace) -> Any:
    if not os.environ.get("RANK_LOG_DIR"):
        return _ProcessGroupCleanupContext(nullcontext())
    path = Path(os.environ["RANK_LOG_DIR"])
    path.mkdir(parents=True, exist_ok=True)
    stream = (path / f"rank-{_rank()}-{args.stage}.log").open("a", encoding="utf-8")

    class Tee:
        def __init__(self, console: Any) -> None:
            self.console = console

        def write(self, value: str) -> int:
            self.console.write(value)
            stream.write(value)
            self.console.flush()
            stream.flush()
            return len(value)

        def flush(self) -> None:
            self.console.flush()
            stream.flush()

        def fileno(self) -> int:
            return self.console.fileno()

    class Context:
        def __enter__(self):
            self.old_out, self.old_err = sys.stdout, sys.stderr
            sys.stdout = Tee(sys.__stdout__)
            sys.stderr = Tee(sys.__stderr__)
            return self

        def __exit__(self, *_):
            sys.stdout, sys.stderr = self.old_out, self.old_err
            stream.close()

    return _ProcessGroupCleanupContext(Context())


def _target_modules(model: Any, family: str) -> list[str]:
    candidates = (
        ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"]
        if family == "glm4_moe_lite"
        else ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    selected = [name for name in candidates if name in names]
    if not selected:
        raise ValueError(f"no supported attention LoRA modules found for {family}")
    return selected


def _load_tokenizer(config: SparkConfig, transformers: Any) -> Any:
    return transformers.AutoTokenizer.from_pretrained(str(config.model_dir), local_files_only=True, revision=config.model_revision)


def _load_model(config: SparkConfig, torch: Any, transformers: Any, peft: Any, load_tuned: bool = False) -> Any:
    model_path = config.output_dir / "model" if load_tuned and config.finetuning_mode == "full" else config.model_dir
    index_path = model_path / "model.safetensors.index.json"
    if load_tuned and config.finetuning_mode == "full" and config.distributed_backend == "deepspeed":
        # output_dir/model is a native DeepSpeed ZeRO checkpoint (no safetensors index); the
        # trained weights are restored later via deepspeed_load_checkpoint onto this skeleton.
        model_config = transformers.AutoConfig.from_pretrained(
            str(config.model_dir), local_files_only=True, revision=config.model_revision
        )
        model = transformers.AutoModelForCausalLM.from_config(model_config, dtype=torch.bfloat16)
    elif config.distributed_backend == "deepspeed" and index_path.is_file():
        from safetensors.torch import load_file
        from transformers.integrations.deepspeed import _load_state_dict_into_zero3_model

        model_config = transformers.AutoConfig.from_pretrained(
            str(model_path), local_files_only=True, revision=config.model_revision
        )
        model = transformers.AutoModelForCausalLM.from_config(model_config, dtype=torch.bfloat16)
        shard_names = dict.fromkeys(json.loads(index_path.read_text(encoding="utf-8"))["weight_map"].values())
        for shard_name in shard_names:
            state_dict = load_file(model_path / shard_name, device="cpu")
            errors, _ = _load_state_dict_into_zero3_model(model, state_dict)
            if errors:
                raise RuntimeError("failed to load ZeRO-3 checkpoint shard: " + "; ".join(errors))
            del state_dict
            gc.collect()
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(str(model_path), dtype=torch.bfloat16, local_files_only=True, revision=config.model_revision)
    if load_tuned and config.finetuning_mode == "lora":
        model = peft.PeftModel.from_pretrained(model, str(config.output_dir / "adapter"), is_trainable=False)
    model.config.use_cache = False
    return model


def _parameter_bytes(parameters: list[Any]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for parameter in parameters:
        dtype = str(parameter.dtype)
        totals[dtype] = totals.get(dtype, 0) + parameter.numel() * parameter.element_size()
    return totals


def _optimizer_state(optimizer: Any) -> dict[Any, Any]:
    seen: set[int] = set()
    current = optimizer
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        state = getattr(current, "state", None)
        if isinstance(state, dict) and state:
            return state
        current = getattr(current, "optimizer", None)
    return {}


def _optimizer_state_bytes(optimizer: Any, torch: Any) -> int:
    return sum(value.numel() * value.element_size() for item in _optimizer_state(optimizer).values() for value in item.values() if torch.is_tensor(value))


def _optimizer_state_dtypes(optimizer: Any, torch: Any) -> list[str]:
    return sorted({str(value.dtype) for item in _optimizer_state(optimizer).values() for value in item.values() if torch.is_tensor(value)})


def _sft_config_kwargs(config: SparkConfig, args: argparse.Namespace) -> dict[str, Any]:
    """Build Trainer arguments before model load so sharded init can take effect."""

    use_gradient_checkpointing = config.distributed_backend != "fsdp2"
    kwargs: dict[str, Any] = {
        "output_dir": str(config.output_dir / "checkpoints"),
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "bf16": True,
        "gradient_checkpointing": use_gradient_checkpointing,
        "gradient_checkpointing_kwargs": (
            {"use_reentrant": False} if use_gradient_checkpointing else None
        ),
        "max_length": config.max_length,
        "completion_only_loss": True,
        "packing": False,
        "eval_strategy": "no",
        "save_strategy": "no",
        "logging_steps": 1,
        "include_num_input_tokens_seen": True,
        "report_to": "none",
        "seed": args.seed,
        "data_seed": args.seed,
        "optim": "sgd" if args.optimizer == "sgd" else "adamw_torch",
        "ddp_backend": "nccl",
    }
    if config.distributed_backend == "ddp":
        kwargs["ddp_find_unused_parameters"] = False
    elif config.distributed_backend == "fsdp2":
        kwargs.update(
            fsdp=True,
            fsdp_config={
                "version": 2,
                "reshard_after_forward": True,
                "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                "activation_checkpointing": True,
                "cpu_ram_efficient_loading": True,
                "state_dict_type": "SHARDED_STATE_DICT",
            },
        )
    else:
        kwargs["deepspeed"] = str(config.deepspeed_config)
    return kwargs


def _estimated_optimizer_state_bytes(parameters: list[Any], optimizer: str) -> int:
    if optimizer == "sgd":
        return 0
    if optimizer == "adamw":
        return sum(2 * parameter.numel() * parameter.element_size() for parameter in parameters)
    raise ValueError(f"unknown optimizer: {optimizer}")


def _uses_deepspeed_nvme(config: SparkConfig) -> bool:
    if config.distributed_backend != "deepspeed" or config.deepspeed_config is None:
        return False
    zero = json.loads(config.deepspeed_config.read_text(encoding="utf-8")).get("zero_optimization", {})
    return any(zero.get(name, {}).get("device") == "nvme" for name in ("offload_param", "offload_optimizer"))


def _sample_trainable_parameters(trainable_named: list[tuple[str, Any]], limit: int = 8) -> list[tuple[str, Any]]:
    """Prefer immediately-updatable LoRA B weights, then full-model output/norm weights."""

    priority_tokens = ("lora_b", "norm", "lm_head", "output")
    selected: list[tuple[str, Any]] = []
    selected_names: set[str] = set()
    # Preserve one representative from every applicable class before filling
    # the bounded sample with additional candidates.
    for token in priority_tokens:
        for name, parameter in trainable_named:
            if token in name.lower() and name not in selected_names:
                selected.append((name, parameter))
                selected_names.add(name)
                break
        if len(selected) == limit:
            return selected
    for token in priority_tokens:
        for name, parameter in trainable_named:
            if token in name.lower() and name not in selected_names:
                selected.append((name, parameter))
                selected_names.add(name)
                if len(selected) == limit:
                    return selected
    for name, parameter in trainable_named:
        if name not in selected_names:
            selected.append((name, parameter))
            selected_names.add(name)
            if len(selected) == limit:
                break
    return selected


def main() -> None:
    args = parse_args()
    config = SparkConfig(
        model_id=args.model_id, model_dir=args.model_dir, model_revision=args.model_revision,
        dataset_id=args.dataset_id, dataset_revision=args.dataset_revision, data_dir=args.data_dir,
        output_dir=args.output_dir, stage=args.stage, finetuning_mode=args.finetuning_mode,
        optimizer=args.optimizer, learning_rate=args.learning_rate, max_steps=args.max_steps,
        max_length=args.max_length, gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=args.seed, distributed_backend=args.distributed_backend,
        deepspeed_config=args.deepspeed_config,
        train_samples=args.train_samples, eval_samples=args.eval_samples,
    )
    family = validate_config(config)
    with _write_log(args):
        import torch
        import transformers
        import trl
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model
        from trl import SFTConfig, SFTTrainer
        from trl_lab.spark_data import prepare_prompt_completion_data

        torch.manual_seed(args.seed)
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Spark path requires a CUDA GPU with BF16 support")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank < 0 or local_rank >= torch.cuda.device_count():
            raise RuntimeError(f"LOCAL_RANK={local_rank} is outside visible CUDA device range")
        torch.cuda.set_device(local_rank)
        training_args = SFTConfig(**_sft_config_kwargs(config, args))
        tokenizer = _load_tokenizer(config, transformers)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        prepared_dir = config.output_dir / f"prepared-data-rank-{_rank()}"
        prepared_paths, data_meta = prepare_prompt_completion_data(
            config.data_dir,
            prepared_dir,
            tokenizer,
            config.dataset_id,
            config.dataset_revision,
            config.max_length,
            config.train_samples,
            config.eval_samples,
        )
        datasets = load_dataset(
            "json",
            data_files={name: str(path) for name, path in prepared_paths.items()},
            cache_dir=str(prepared_dir / "hf-cache"),
        )
        model = _load_model(config, torch, transformers, __import__("peft"), load_tuned=args.stage == "tuned")
        if training_args.gradient_checkpointing:
            if not hasattr(model, "gradient_checkpointing_enable"):
                raise RuntimeError("model does not expose gradient_checkpointing_enable")
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if args.finetuning_mode == "lora" and args.stage != "tuned":
            model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM", target_modules=_target_modules(model, family)))
        if args.stage == "base":
            model.requires_grad_(False)
        trainable_named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        trainable = [parameter for _, parameter in trainable_named]
        all_parameters = list(model.parameters())
        total_parameter_count = sum(parameter.numel() for parameter in all_parameters)
        trainable_parameter_count = sum(parameter.numel() for parameter in trainable)
        sample_candidates = _sample_trainable_parameters(trainable_named) if config.distributed_backend == "ddp" else []
        before_sample = {name: parameter.detach().reshape(-1)[:16].float().cpu().clone() for name, parameter in sample_candidates} if args.stage == "train" else {}
        from trl_lab.observatory import make_trl_callback
        callback = make_trl_callback()
        trainer_kwargs = dict(model=model, args=training_args, train_dataset=datasets["train"], eval_dataset=datasets["validation"], processing_class=tokenizer)
        if callback is not None:
            trainer_kwargs["callbacks"] = [callback]
        trainer = SFTTrainer(**trainer_kwargs)
        train_metrics: dict[str, Any] = {}
        train_seconds = None
        if args.stage in {"base", "tuned"}:
            if args.stage == "tuned" and config.distributed_backend == "deepspeed" and args.finetuning_mode == "full":
                from transformers.integrations.deepspeed import deepspeed_load_checkpoint

                train_dataloader = trainer.get_train_dataloader()
                trainer._prepare_for_training(max_steps=1, train_dataloader=train_dataloader, resume_from_checkpoint=None)
                deepspeed_load_checkpoint(trainer.model_wrapped, str(config.output_dir / "model"), load_module_strict=True)
            evaluation = trainer.evaluate()
            update_count = 0
        else:
            started = time.perf_counter()
            result = trainer.train()
            evaluation = (
                {"skipped": True, "reason": "DeepSpeed NVMe swap buffers are terminal after the optimizer step"}
                if _uses_deepspeed_nvme(config)
                else trainer.evaluate()
            )
            trainer.save_model(str(config.output_dir / "adapter" if args.finetuning_mode == "lora" else config.output_dir / "model"))
            train_metrics = dict(result.metrics)
            update_count = int(result.global_step)
            train_seconds = time.perf_counter() - started
        optimizer_instance = getattr(trainer, "optimizer", None) if args.stage == "train" else None
        after_sample = {name: parameter.detach().reshape(-1)[:16].float().cpu() for name, parameter in sample_candidates} if args.stage == "train" else {}
        update_deltas = {name: float((after_sample[name] - before_sample[name]).abs().max().item()) for name in before_sample}
        eval_loss = evaluation.get("eval_loss")
        evaluation_completed = not evaluation.get("skipped", False)
        finite_eval_loss = isinstance(eval_loss, (int, float)) and math.isfinite(float(eval_loss)) if evaluation_completed else None
        train_loss = train_metrics.get("train_loss")
        finite_train_loss = args.stage != "train" or (isinstance(train_loss, (int, float)) and math.isfinite(float(train_loss)))
        sampled_nonzero_update = any(value > 0.0 for value in update_deltas.values()) if args.stage == "train" else None
        if _rank() == 0:
            config.output_dir.mkdir(parents=True, exist_ok=True)
            allocated = None
            if torch.cuda.is_available():
                allocated = torch.cuda.max_memory_allocated() / 1024**3
                reserved = torch.cuda.max_memory_reserved() / 1024**3
            else:
                reserved = None
            optimizer_dtypes = []
            optimizer_actual_bytes = None
            if optimizer_instance is not None:
                optimizer_dtypes = _optimizer_state_dtypes(optimizer_instance, torch)
                optimizer_actual_bytes = _optimizer_state_bytes(optimizer_instance, torch)
            validation_errors = []
            if finite_eval_loss is False:
                validation_errors.append("evaluation loss is not finite")
            if not finite_train_loss:
                validation_errors.append("training loss is not finite")
            if args.stage == "train" and update_count < 1:
                validation_errors.append("trainer reported no optimizer steps")
            if args.stage == "train" and config.distributed_backend == "ddp" and not sampled_nonzero_update:
                validation_errors.append("no sampled trainable parameter changed")
            summary = {"stage": args.stage, "rank": _rank(), "world_size": int(os.environ.get("WORLD_SIZE", "1")), "node_rank": int(os.environ.get("NODE_RANK", "0")), "distributed_backend": config.distributed_backend, "deepspeed_config": str(config.deepspeed_config) if config.deepspeed_config else None, "model_id": args.model_id, "model_family": family, "model_revision": args.model_revision, "model_snapshot_evidence": model_snapshot_evidence(config.model_dir), "dataset_id": args.dataset_id, "dataset_revision": args.dataset_revision, "finetuning_mode": args.finetuning_mode, "optimizer": args.optimizer, "learning_rate": args.learning_rate, "precision": "bf16", "gradient_checkpointing": {"enabled": training_args.gradient_checkpointing, "use_reentrant": False if training_args.gradient_checkpointing else None, "fsdp_activation_checkpointing": config.distributed_backend == "fsdp2"}, "optimizer_state_estimate_bytes": _estimated_optimizer_state_bytes(trainable, args.optimizer) if args.stage == "train" else None, "optimizer_state_actual_bytes": optimizer_actual_bytes, "optimizer_state_scope": "rank-local shard" if config.distributed_backend != "ddp" else "replicated rank-local optimizer", "optimizer_state_dtypes": optimizer_dtypes, "memory_components_scope": "pre-wrap logical model view; not a rank-local sharded allocation" if config.distributed_backend != "ddp" else "replicated model view", "memory_components_bytes": {"parameters_by_dtype": _parameter_bytes(all_parameters), "trainable_gradients_by_dtype": _parameter_bytes(trainable), "optimizer_state_estimate": _estimated_optimizer_state_bytes(trainable, args.optimizer) if args.stage == "train" else 0, "optimizer_state_actual": optimizer_actual_bytes}, "max_steps": args.max_steps, "actual_optimizer_steps": update_count, "train_seconds": train_seconds, "train_metrics": train_metrics, "trainable_parameter_count": trainable_parameter_count, "total_parameter_count": total_parameter_count, "trainable_parameter_fraction": trainable_parameter_count / total_parameter_count if total_parameter_count else 0.0, "sampled_parameter_names": [name for name, _ in sample_candidates], "sampled_parameter_update_max_abs": update_deltas, "peak_cuda_memory_allocated_gib": allocated, "peak_cuda_memory_reserved_gib": reserved, "data": data_meta, "evaluation": evaluation, "validation": {"summary_writer_rank": 0, "evaluation_completed": evaluation_completed, "finite_eval_loss": finite_eval_loss, "finite_train_loss": finite_train_loss, "supervision_policy": "native chat-template prompt plus final assistant completion and EOS", "parameter_update_evidence": "sampled parameter delta" if config.distributed_backend == "ddp" else "optimizer steps only; sharded parameter delta not collected", "sampled_nonzero_update": sampled_nonzero_update, "training_result_verified": not validation_errors, "failure_reasons": validation_errors}}
            (config.output_dir / f"summary-{args.stage}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
        if finite_eval_loss is False or not finite_train_loss or (args.stage == "train" and update_count < 1) or (args.stage == "train" and config.distributed_backend == "ddp" and not any(value > 0.0 for value in update_deltas.values())):
            raise RuntimeError("training result is unverified: non-finite loss or no observed trainable update")


if __name__ == "__main__":
    main()
