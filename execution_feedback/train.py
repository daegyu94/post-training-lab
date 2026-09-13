"""TRL entry point for execution-filtered SFT and DPO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time


def dpo_needs_precomputed_ref_logps(is_adapter: bool, lora_r: int) -> bool:
    """True for full fine-tuning only. With ref_model=None, DPOTrainer keeps a
    second full copy of a non-PEFT model resident for the whole run; PEFT models
    already avoid that cheaply via adapter-disable, so they don't need this."""
    return not (is_adapter or bool(lora_r))


def single_process_device_map(world_size: int) -> str | None:
    """"auto" outside a distributed launch so a large model loads shard-by-shard
    straight into the one visible GPU instead of fully materializing on host RAM
    first. Under torchrun/accelerate launch (world_size > 1) each rank already
    has its own assigned device, and "auto" would wrongly try to shard across
    every GPU visible to that rank, so this only applies to a single process."""
    return "auto" if world_size == 1 else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("sft", "dpo"), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--eval-file", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--lora-r", type=int, default=0)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--save-checkpoint", action="store_true")
    args = parser.parse_args()
    if args.max_steps < 1 or args.per_device_batch_size < 1 or args.gradient_accumulation_steps < 1:
        parser.error("step and batch values must be positive")

    import os
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    is_adapter = (args.model_dir / "adapter_config.json").is_file()
    device_map = single_process_device_map(int(os.environ.get("WORLD_SIZE", "1")))
    if is_adapter:
        if args.lora_r:
            parser.error("--lora-r cannot create a second adapter on an adapter checkpoint")
        from peft import AutoPeftModelForCausalLM
        model = AutoPeftModelForCausalLM.from_pretrained(args.model_dir, local_files_only=True, torch_dtype=torch.bfloat16, is_trainable=True, device_map=device_map)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_dir, local_files_only=True, torch_dtype=torch.bfloat16, device_map=device_map)
    if args.lora_r and not is_adapter:
        from peft import LoraConfig, get_peft_model
        # PEFT's architecture->target_modules default mapping doesn't cover this
        # repo's MoE models (Qwen3-30B-A3B's fused-expert-parameter MoE, GLM's MLA
        # attention), so leaving target_modules unset raises "No target_modules
        # passed but also no target_parameters found" before training starts.
        # all-linear auto-detects every real Linear (and, on this PEFT version,
        # fused-expert) layer regardless of architecture.
        model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM", target_modules="all-linear"))
    files = {"train": str(args.train_file)}
    if args.eval_file:
        files["validation"] = str(args.eval_file)
    dataset = load_dataset("json", data_files=files)
    common = dict(
        output_dir=str(args.output_dir), max_steps=args.max_steps, learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=True, gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
        # A Trainer-native checkpoint duplicates trainer.save_model()'s output
        # below plus a full optimizer state (observed: ~2x the adapter size in
        # fp32 moment buffers, e.g. 8GB optimizer.pt next to a 4GB LoRA adapter
        # on a 30B model). It's only useful for --resume-from-checkpoint after a
        # crash, so it's opt-in rather than paid on every successful run.
        seed=args.seed, logging_steps=1,
        save_strategy="steps" if args.save_checkpoint else "no", save_steps=args.max_steps,
        eval_strategy="steps" if args.eval_file else "no", eval_steps=args.max_steps if args.eval_file else None,
        report_to="none", remove_unused_columns=False, include_num_input_tokens_seen=True,
    )
    if args.mode == "sft":
        from trl import SFTConfig, SFTTrainer
        # SFTConfig defaults to loss_type="chunked_nll", which patches the model's
        # forward assuming the output embedding's forward is a plain bound method.
        # A CPU-offloaded module (device_map="auto" placing part of a large model
        # on CPU) has its forward wrapped by accelerate as a functools.partial
        # instead, so the patch crashes with "'functools.partial' object has no
        # attribute '__func__'" before training starts. Observed directly loading
        # Qwen3-30B-A3B under CPU offload. "nll" is the same loss unchunked, with
        # no forward patching.
        training_args = SFTConfig(max_length=args.max_length, loss_type="nll", **common)
        trainer = SFTTrainer(model=model, args=training_args, train_dataset=dataset["train"], eval_dataset=dataset.get("validation"), processing_class=tokenizer)
    else:
        from trl import DPOConfig, DPOTrainer
        # With ref_model=None and no PEFT adapter, DPOTrainer reloads the whole
        # base model a second time to keep as a frozen reference (dpo_trainer.py:
        # "Reference model" branch, ref_model_init_kwargs -> create_model_from_path),
        # doubling resident weights for the run's full duration. On a 30B model
        # that is the difference between fitting and not. precompute_ref_log_probs
        # instead runs the model once over the dataset before training (using its
        # own pre-update weights as the reference) and never keeps a second copy
        # (trainer stays ref_model=None; see _precompute_ref_logps). PEFT already
        # avoids the second copy cheaply via adapter-disable, so this only needs
        # to be forced on for true full fine-tuning.
        training_args = DPOConfig(
            max_length=args.max_length, beta=args.beta,
            precompute_ref_log_probs=dpo_needs_precomputed_ref_logps(is_adapter, args.lora_r), **common,
        )
        trainer = DPOTrainer(model=model, ref_model=None, args=training_args, train_dataset=dataset["train"], eval_dataset=dataset.get("validation"), processing_class=tokenizer)
    started = time.perf_counter()
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(args.output_dir / "model"))
    summary = {
        "mode": args.mode, "base_model_dir": str(args.model_dir), "train_file": str(args.train_file),
        "eval_file": str(args.eval_file) if args.eval_file else None, "max_steps": args.max_steps,
        "learning_rate": args.learning_rate, "per_device_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps, "beta": args.beta if args.mode == "dpo" else None,
        "lora_r": args.lora_r, "continued_adapter": is_adapter, "seed": args.seed, "duration_seconds": time.perf_counter() - started,
        "num_input_tokens_seen": getattr(trainer.state, "num_input_tokens_seen", None),
        "metrics": dict(result.metrics), "world_size": int(os.environ.get("WORLD_SIZE", "1")),
    }
    if trainer.is_world_process_zero():
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "execution_feedback_training.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

