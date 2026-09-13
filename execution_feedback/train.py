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
    args = parser.parse_args()
    if args.max_steps < 1 or args.per_device_batch_size < 1 or args.gradient_accumulation_steps < 1:
        parser.error("step and batch values must be positive")

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    is_adapter = (args.model_dir / "adapter_config.json").is_file()
    if is_adapter:
        if args.lora_r:
            parser.error("--lora-r cannot create a second adapter on an adapter checkpoint")
        from peft import AutoPeftModelForCausalLM
        model = AutoPeftModelForCausalLM.from_pretrained(args.model_dir, local_files_only=True, torch_dtype=torch.bfloat16, is_trainable=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_dir, local_files_only=True, torch_dtype=torch.bfloat16)
    if args.lora_r and not is_adapter:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))
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
        seed=args.seed, logging_steps=1, save_strategy="steps", save_steps=args.max_steps,
        eval_strategy="steps" if args.eval_file else "no", eval_steps=args.max_steps if args.eval_file else None,
        report_to="none", remove_unused_columns=False, include_num_input_tokens_seen=True,
    )
    if args.mode == "sft":
        from trl import SFTConfig, SFTTrainer
        training_args = SFTConfig(max_length=args.max_length, **common)
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
        "metrics": dict(result.metrics), "world_size": int(__import__("os").environ.get("WORLD_SIZE", "1")),
    }
    if trainer.is_world_process_zero():
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "execution_feedback_training.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

