"""Generate one or more code candidates from a local checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from execution_feedback.common import index_tasks, sha256, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--num-candidates", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--revision")
    args = parser.parse_args()
    if args.num_candidates < 1:
        parser.error("--num-candidates must be positive")

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
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        for ordinal in range(args.num_candidates):
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=args.temperature > 0, temperature=args.temperature if args.temperature > 0 else None, top_p=args.top_p if args.temperature > 0 else None, pad_token_id=tokenizer.eos_token_id)
            response = tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            rows.append({"task_id": task["task_id"], "candidate_id": str(ordinal), "response": response, "generation": {"model_dir": str(args.model_dir), "split": args.split, "seed": args.seed, "temperature": args.temperature, "top_p": args.top_p, "max_new_tokens": args.max_new_tokens}})
    write_jsonl(args.output, rows)
    manifest = {"tasks_sha256": sha256(args.tasks), "model_dir": str(args.model_dir), "split": args.split, "task_count": len(tasks), "candidate_count": len(rows), "duration_seconds": time.perf_counter() - started, "generation": {"seed": args.seed, "temperature": args.temperature, "top_p": args.top_p, "max_new_tokens": args.max_new_tokens}}
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"task_count": len(tasks), "candidate_count": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
