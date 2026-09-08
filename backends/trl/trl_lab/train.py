"""Train and evaluate Qwen2.5-14B-Instruct with NF4 QLoRA and TRL."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import transformers
import trl
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from trl import SFTConfig, SFTTrainer

from trl_lab.data import QWEN_ASSISTANT_MASK_TEMPLATE, load_sft_data


DEFAULT_PROMPTS = [
    "Explain gradient accumulation in three concise bullet points.",
    "Give two practical tips for debugging an out-of-memory error during LLM training.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--dataset", default="HuggingFaceH4/ultrachat_200k")
    parser.add_argument("--dataset-parquet-dir", type=Path)
    parser.add_argument("--dataset-jsonl-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/qwen2.5-14b-qlora"))
    parser.add_argument("--train-samples", type=int, default=128)
    parser.add_argument("--eval-samples", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _quantization_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def _generate(model: Any, tokenizer: Any, prompts: list[str]) -> list[dict[str, str]]:
    outputs: list[dict[str, str]] = []
    model.eval()
    for prompt in prompts:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=96,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        answer = tokenizer.decode(generated[0, inputs.input_ids.shape[1] :], skip_special_tokens=True)
        outputs.append({"prompt": prompt, "answer": answer.strip()})
    return outputs


def _finite_perplexity(loss: float) -> float:
    return math.exp(loss) if loss < 20 else float("inf")


_LEVEL_COLORS = {
    "INFO": "\033[1;36m",
    "WARNING": "\033[1;33m",
    "ERROR": "\033[1;31m",
}
_RESET_COLOR = "\033[0m"


def _log(level: str, message: str) -> None:
    stream = sys.stderr if level == "ERROR" else sys.stdout
    label = f"[{level}]"
    if stream.isatty() and not os.environ.get("NO_COLOR"):
        label = f"{_LEVEL_COLORS[level]}{label}{_RESET_COLOR}"
    print(f"{label} {message}", file=stream, flush=True)


def _log_stage(stage: str, detail: str) -> None:
    _log("INFO", f"{stage}: {detail}")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for the 14B QLoRA exercise")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This configuration requires a GPU with BF16 support")
    if args.train_samples < 1 or args.eval_samples < 1 or args.max_steps < 1:
        raise ValueError("train-samples, eval-samples, and max-steps must be positive")

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _log_stage(
        "Stage 1/6",
        f"Loading dataset from {args.dataset_jsonl_dir or args.dataset_parquet_dir or args.dataset}",
    )
    train_dataset, eval_dataset = load_sft_data(
        args.dataset,
        args.train_samples,
        args.eval_samples,
        args.seed,
        args.dataset_parquet_dir,
        args.dataset_jsonl_dir,
    )

    _log_stage(
        "Stage 1/6",
        f"Dataset ready: train={len(train_dataset)}, evaluation={len(eval_dataset)}",
    )
    _log_stage("Stage 2/6", "Loading tokenizer and 4-bit Qwen base model")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    tokenizer.chat_template = QWEN_ASSISTANT_MASK_TEMPLATE
    tokenizer.pad_token = tokenizer.eos_token
    mask_probe = tokenizer.apply_chat_template(
        [{"role": "user", "content": "probe"}, {"role": "assistant", "content": "answer"}],
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    if sum(mask_probe["assistant_masks"]) == 0:
        raise RuntimeError("The chat template did not produce an assistant token mask")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=_quantization_config(),
        dtype=torch.bfloat16,
        device_map={"": 0},
        local_files_only=args.local_files_only,
    )
    model.config.use_cache = False
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    training_args = SFTConfig(
        output_dir=str(args.output_dir / "checkpoints"),
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.max_length,
        assistant_only_loss=True,
        packing=False,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=args.max_steps,
        save_total_limit=1,
        logging_steps=1,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    _log_stage("Stage 3/6", "Evaluating the base model on held-out conversations")
    disable_adapter = getattr(trainer.model, "disable_adapter", None)
    adapter_context = disable_adapter() if disable_adapter else nullcontext()
    with adapter_context:
        base_eval = trainer.evaluate(metric_key_prefix="base_eval")
        base_generations = _generate(trainer.model, tokenizer, DEFAULT_PROMPTS)

    torch.cuda.reset_peak_memory_stats()
    _log_stage("Stage 4/6", f"Training the QLoRA adapter for {args.max_steps} optimizer steps")
    started = time.perf_counter()
    train_result = trainer.train()
    train_seconds = time.perf_counter() - started
    _log_stage("Stage 5/6", "Evaluating the tuned model and generating comparison responses")
    tuned_eval = trainer.evaluate(metric_key_prefix="tuned_eval")
    tuned_generations = _generate(trainer.model, tokenizer, DEFAULT_PROMPTS)
    peak_memory_gib = torch.cuda.max_memory_allocated() / 1024**3

    _log_stage("Stage 6/6", f"Saving adapter, checkpoint, and summary under {args.output_dir}")
    adapter_dir = args.output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(adapter_dir)
    base_loss = float(base_eval["base_eval_loss"])
    tuned_loss = float(tuned_eval["tuned_eval_loss"])
    summary = {
        "configuration": {
            "model": args.model,
            "dataset": args.dataset,
            "dataset_jsonl_dir": str(args.dataset_jsonl_dir) if args.dataset_jsonl_dir else None,
            "train_samples": args.train_samples,
            "eval_samples": args.eval_samples,
            "max_steps": args.max_steps,
            "max_length": args.max_length,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "effective_train_samples": len(trainer.train_dataset),
            "effective_eval_samples": len(trainer.eval_dataset),
            "seed": args.seed,
        },
        "quality": {
            "base_eval_loss": base_loss,
            "tuned_eval_loss": tuned_loss,
            "loss_change_percent": 100 * (tuned_loss - base_loss) / base_loss,
            "base_perplexity": _finite_perplexity(base_loss),
            "tuned_perplexity": _finite_perplexity(tuned_loss),
        },
        "performance": {
            "train_seconds": train_seconds,
            "steps_per_second": train_result.metrics.get("train_steps_per_second"),
            "samples_per_second": train_result.metrics.get("train_samples_per_second"),
            "peak_allocated_gpu_memory_gib": peak_memory_gib,
        },
        "generations": {"base": base_generations, "tuned": tuned_generations},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "trl": trl.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not set"),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _log_stage("Complete", f"Summary written to {args.output_dir / 'summary.json'}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        _log("ERROR", f"{type(error).__name__}: {error}")
        raise
