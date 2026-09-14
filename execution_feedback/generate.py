"""Generate one or more code candidates from a local checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

from execution_feedback.common import index_tasks, sha256, write_jsonl


def candidate_budgets(count: int, max_new_tokens: int, truncated_count: int, truncated_max_new_tokens: int) -> list[tuple[str, int]]:
    return (
        [(str(ordinal), max_new_tokens) for ordinal in range(count)]
        + [(f"truncated-{ordinal}", truncated_max_new_tokens) for ordinal in range(truncated_count)]
    )


def generation_batch(inputs: dict[str, Any], count: int, do_sample: bool) -> tuple[dict[str, Any], dict[str, int]]:
    if do_sample:
        return inputs, {"num_return_sequences": count}
    return ({name: value.repeat_interleave(count, dim=0) for name, value in inputs.items()}, {})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--num-candidates", type=int, default=1)
    parser.add_argument("--truncated-candidates", type=int, default=0)
    parser.add_argument("--truncated-max-new-tokens", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--revision")
    parser.add_argument("--eval-file", type=Path)
    parser.add_argument("--eval-output", type=Path)
    parser.add_argument("--eval-max-length", type=int, default=2048)
    args = parser.parse_args()
    if args.num_candidates < 1 or args.truncated_candidates < 0 or args.truncated_max_new_tokens < 1:
        parser.error("candidate counts and token limits must be positive")
    if bool(args.eval_file) != bool(args.eval_output):
        parser.error("--eval-file and --eval-output must be used together")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, revision=args.revision, local_files_only=True)
    if (args.model_dir / "adapter_config.json").is_file():
        from peft import AutoPeftModelForCausalLM
        model = AutoPeftModelForCausalLM.from_pretrained(args.model_dir, local_files_only=True, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_dir, revision=args.revision, local_files_only=True, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, device_map="auto")
    model.eval()
    tasks = [task for task in index_tasks(args.tasks).values() if task["split"] == args.split]
    rows = []
    started = time.perf_counter()
    for task in tasks:
        messages = [{"role": "user", "content": task["prompt"]}]
        # Reasoning models (e.g. Qwen3) emit a <think>...</think> block before any
        # code; on a bounded --max-new-tokens budget that reasoning can consume the
        # whole budget and leave zero code tokens, failing every candidate. Observed
        # directly: 12/12 real candidates from Qwen3-30B-A3B were bare truncated
        # <think> text at max-new-tokens=64. enable_thinking=False pre-closes the
        # think block in the prompt itself; templates that don't support it ignore
        # the extra kwarg.
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        # Candidates for the same task share one prompt, so every candidate at a
        # given max_new_tokens budget is one num_return_sequences batch instead of
        # a separate model.generate() call. Autoregressive decode is memory
        # bandwidth-bound (each step reloads the full weight set regardless of
        # batch size), so batching candidates is close to free throughput instead
        # of a linear multiplier in wall time.
        budgets_by_length: dict[int, list[str]] = {}
        for candidate_id, max_new_tokens in candidate_budgets(
            args.num_candidates, args.max_new_tokens,
            args.truncated_candidates, args.truncated_max_new_tokens,
        ):
            budgets_by_length.setdefault(max_new_tokens, []).append(candidate_id)
        input_length = inputs["input_ids"].shape[1]
        for max_new_tokens, candidate_ids in budgets_by_length.items():
            do_sample = args.temperature > 0
            batch_inputs, batch_kwargs = generation_batch(inputs, len(candidate_ids), do_sample)
            with torch.inference_mode():
                generated = model.generate(**batch_inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=args.temperature if do_sample else None, top_p=args.top_p if do_sample else None, pad_token_id=tokenizer.eos_token_id, **batch_kwargs)
            for row_index, candidate_id in enumerate(candidate_ids):
                response = tokenizer.decode(generated[row_index, input_length:], skip_special_tokens=True)
                rows.append({"task_id": task["task_id"], "candidate_id": candidate_id, "response": response, "generation": {"model_dir": str(args.model_dir), "split": args.split, "seed": args.seed, "temperature": args.temperature, "top_p": args.top_p, "max_new_tokens": max_new_tokens}})
    write_jsonl(args.output, rows)
    manifest = {"tasks_sha256": sha256(args.tasks), "model_dir": str(args.model_dir), "split": args.split, "task_count": len(tasks), "candidate_count": len(rows), "duration_seconds": time.perf_counter() - started, "generation": {"seed": args.seed, "temperature": args.temperature, "top_p": args.top_p, "max_new_tokens": args.max_new_tokens, "truncated_candidates": args.truncated_candidates, "truncated_max_new_tokens": args.truncated_max_new_tokens}}
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.eval_file:
        from datasets import load_dataset
        from trl import SFTConfig, SFTTrainer

        evaluation = load_dataset("json", data_files={"validation": str(args.eval_file)})["validation"]
        trainer = SFTTrainer(
            model=model,
            args=SFTConfig(
                output_dir=str(args.eval_output.parent / ".eval"), max_length=args.eval_max_length,
                loss_type="nll", per_device_eval_batch_size=1, bf16=torch.cuda.is_available(),
                gradient_checkpointing=False, report_to="none", remove_unused_columns=False,
            ),
            train_dataset=evaluation, eval_dataset=evaluation, processing_class=tokenizer,
        )
        metrics = trainer.evaluate()
        args.eval_output.parent.mkdir(parents=True, exist_ok=True)
        args.eval_output.write_text(json.dumps({
            "model_dir": str(args.model_dir), "eval_file": str(args.eval_file),
            "eval_file_sha256": sha256(args.eval_file), "task_count": len(evaluation),
            "eval_loss": metrics["eval_loss"], "metrics": metrics,
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"task_count": len(tasks), "candidate_count": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
