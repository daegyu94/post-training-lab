"""Train a LoRA adapter with Docker-scored GRPO on execution-feedback tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from execution_feedback.common import read_jsonl
from execution_feedback.evaluate import evaluate_candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--max-completion-length", type=int, default=128)
    parser.add_argument("--image", default="python:3.12-slim")
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    args = parser.parse_args()
    if args.max_steps < 1 or args.num_generations < 2 or args.max_completion_length < 1:
        parser.error("steps, generations, and completion length must be positive; generations must be at least two")
    if not (args.model_dir / "adapter_config.json").is_file():
        parser.error("--model-dir must be a trainable LoRA adapter")

    import torch
    from datasets import Dataset
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    tasks = list(read_jsonl(args.tasks))
    if not tasks:
        parser.error("--tasks is empty")
    rows = [{"prompt": task["prompt"], "task": json.dumps(task)} for task in tasks if task["split"] == "train"]
    if not rows:
        parser.error("--tasks has no train examples")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.model_dir, local_files_only=True, torch_dtype=torch.bfloat16, is_trainable=True, device_map="auto"
    )

    def execution_reward(completions: list[str], task: list[str], **_: object) -> list[float]:
        scores = []
        for response, raw_task in zip(completions, task, strict=True):
            result = evaluate_candidate(json.loads(raw_task), {"candidate_id": "rlvr", "response": response}, image=args.image, timeout_seconds=args.timeout_seconds)
            if result["status"] == "infra_error":
                raise RuntimeError("Docker evaluator infrastructure failed: " + "; ".join(result["failures"]))
            scores.append(1.0 if result["status"] == "pass" else 0.0)
        return scores

    training_args = GRPOConfig(
        output_dir=str(args.output_dir), max_steps=args.max_steps, learning_rate=1e-6,
        per_device_train_batch_size=1, gradient_accumulation_steps=1, bf16=True,
        num_generations=args.num_generations, generation_batch_size=args.num_generations,
        max_completion_length=args.max_completion_length,
        report_to="none", logging_steps=1, save_strategy="steps", save_steps=args.max_steps,
        remove_unused_columns=False, use_vllm=False,
    )
    trainer = GRPOTrainer(
        model=model, reward_funcs=execution_reward, args=training_args,
        train_dataset=Dataset.from_list(rows), processing_class=tokenizer,
    )
    result = trainer.train()
    trainer.save_model(str(args.output_dir / "model"))
    print(json.dumps({"max_steps": args.max_steps, "task_count": len(rows), "metrics": result.metrics}, ensure_ascii=False))


if __name__ == "__main__":
    main()
