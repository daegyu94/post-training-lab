"""Native-template prompt/completion preparation for Spark SFT."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from sft_lab.spark_config import validate_dataset_manifest


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def render_prompt_completion(row: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("messages must contain a final assistant turn")
    final = messages[-1]
    if not isinstance(final, dict) or final.get("role") != "assistant":
        raise ValueError("SFT row must end with assistant")
    if final.get("tool_calls") is not None or final.get("function_call") is not None:
        raise ValueError("final assistant structured tool calls are unsupported")
    prompt_messages = messages[:-1]
    for message in prompt_messages:
        if not isinstance(message, dict) or message.get("tool_calls") is not None or message.get("function_call") is not None:
            raise ValueError("prompt contains unsupported structured tool calls")
        role = _text(message.get("role"), "message.role")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported message role: {role}")
        _text(message.get("content"), "message.content")
    if not any(message.get("role") == "user" for message in prompt_messages if isinstance(message, dict)):
        raise ValueError("SFT prompt must contain a user turn")
    prompt = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    eos = _text(getattr(tokenizer, "eos_token", None), "tokenizer.eos_token")
    completion = _text(final.get("content"), "assistant.content") + eos
    prompt_tokens = _token_ids(tokenizer, prompt)
    combined_tokens = _token_ids(tokenizer, prompt + completion)
    if combined_tokens[:len(prompt_tokens)] != prompt_tokens:
        raise ValueError("native prompt tokenization is not a prefix of prompt plus completion")
    supervised_tokens = len(combined_tokens) - len(prompt_tokens)
    if supervised_tokens < 1:
        raise ValueError("final assistant completion has no supervised tokens")
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None and combined_tokens[-1] != eos_id:
        raise ValueError("tokenizer EOS is not the final supervised token")
    prompt_id = row.get("prompt_id")
    if prompt_id is None:
        raise ValueError("prompt_id is required")
    return {
        "prompt": _text(prompt, "rendered prompt"),
        "completion": completion,
        "prompt_id": _text(str(prompt_id), "prompt_id"),
        "supervised_tokens": supervised_tokens,
        "prompt_tokens": len(prompt_tokens),
        "provenance": row.get("provenance", {}),
    }


def load_prompt_completion_data(
    data_dir: Path,
    tokenizer: Any,
    dataset_id: str,
    dataset_revision: str,
    max_length: int,
    train_samples: int | None = None,
    eval_samples: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    manifest = validate_dataset_manifest(data_dir, dataset_id, dataset_revision)
    if max_length < 1:
        raise ValueError("max_length must be positive")
    if train_samples is not None and train_samples < 1:
        raise ValueError("train_samples must be positive when specified")
    if eval_samples is not None and eval_samples < 1:
        raise ValueError("eval_samples must be positive when specified")
    prepared: dict[str, list[dict[str, Any]]] = {}
    for name, filename, limit in (("train", "training.jsonl", train_samples), ("validation", "validation.jsonl", eval_samples)):
        rows = []
        with (data_dir / filename).open(encoding="utf-8") as source:
            for line in source:
                if limit is not None and len(rows) >= limit:
                    break
                row = json.loads(line)
                example = render_prompt_completion(row, tokenizer)
                total_tokens = len(_token_ids(tokenizer, example["prompt"] + example["completion"]))
                if total_tokens > max_length:
                    raise ValueError(f"{name} example {example['prompt_id']} exceeds max_length={max_length}")
                rows.append(example)
        if not rows:
            raise ValueError(f"{name} split is empty")
        if limit is not None and len(rows) != limit:
            raise ValueError(f"{name} split has fewer than requested {limit} examples")
        prepared[name] = rows
    train_ids = {row["prompt_id"] for row in prepared["train"]}
    eval_ids = {row["prompt_id"] for row in prepared["validation"]}
    if train_ids & eval_ids:
        raise ValueError("train and validation prompt IDs overlap")
    return prepared["train"], prepared["validation"], {
        "manifest": manifest,
        "train_count": len(prepared["train"]),
        "eval_count": len(prepared["validation"]),
        "actual_selected_rows": {
            "train": [row["prompt_id"] for row in prepared["train"]],
            "validation": [row["prompt_id"] for row in prepared["validation"]],
        },
        "supervised_tokens": {
            "train": sum(row["supervised_tokens"] for row in prepared["train"]),
            "validation": sum(row["supervised_tokens"] for row in prepared["validation"]),
        },
        "data_sha256": hashlib.sha256((data_dir / "training.jsonl").read_bytes() + (data_dir / "validation.jsonl").read_bytes()).hexdigest(),
    }
