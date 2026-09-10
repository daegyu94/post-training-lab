"""Native-template prompt/completion preparation for Spark SFT."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

from trl_lab.spark_config import validate_dataset_manifest


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


def _sha256_files(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _iter_examples(
    path: Path,
    tokenizer: Any,
    split: str,
    max_length: int,
    limit: int | None,
):
    count = 0
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            if limit is not None and count >= limit:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}") from exc
            example = render_prompt_completion(row, tokenizer)
            total_tokens = example["prompt_tokens"] + example["supervised_tokens"]
            if total_tokens > max_length:
                print(
                    f"[data] skipping {split} example {example['prompt_id']}: "
                    f"{total_tokens} tokens exceeds max_length={max_length}",
                    file=sys.stderr,
                )
                continue
            count += 1
            yield example
    if count == 0:
        raise ValueError(f"{split} split is empty")
    if limit is not None and count != limit:
        raise ValueError(f"{split} split has fewer than requested {limit} examples")


def prepare_prompt_completion_data(
    data_dir: Path,
    output_dir: Path,
    tokenizer: Any,
    dataset_id: str,
    dataset_revision: str,
    max_length: int,
    train_samples: int | None = None,
    eval_samples: int | None = None,
) -> tuple[dict[str, Path], dict[str, Any]]:
    """Stream canonical JSONL into disk-backed Trainer inputs."""

    manifest = validate_dataset_manifest(data_dir, dataset_id, dataset_revision)
    if max_length < 1:
        raise ValueError("max_length must be positive")
    if train_samples is not None and train_samples < 1:
        raise ValueError("train_samples must be positive when specified")
    if eval_samples is not None and eval_samples < 1:
        raise ValueError("eval_samples must be positive when specified")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    counts: dict[str, int] = {}
    supervised_tokens: dict[str, int] = {}
    selected: dict[str, list[str]] = {}
    train_ids: set[str] = set()
    for split, filename, limit in (
        ("train", "training.jsonl", train_samples),
        ("validation", "validation.jsonl", eval_samples),
    ):
        destination = output_dir / filename
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{filename}.{os.getpid()}-", dir=str(output_dir), text=True
        )
        count = 0
        token_count = 0
        ids: set[str] = set()
        id_sample: list[str] = []
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as target:
                for example in _iter_examples(
                    data_dir / filename, tokenizer, split, max_length, limit
                ):
                    prompt_id = example["prompt_id"]
                    if split == "validation" and prompt_id in train_ids:
                        raise ValueError("train and validation prompt IDs overlap")
                    if split == "train":
                        ids.add(prompt_id)
                    if len(id_sample) < 100:
                        id_sample.append(prompt_id)
                    count += 1
                    token_count += example["supervised_tokens"]
                    target.write(
                        json.dumps(
                            {key: example[key] for key in ("prompt", "completion")},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                target.flush()
                os.fsync(target.fileno())
            Path(temporary_name).replace(destination)
        except BaseException:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass
            raise
        paths[split] = destination
        counts[split] = count
        supervised_tokens[split] = token_count
        selected[split] = id_sample
        if split == "train":
            train_ids = ids
    return paths, {
        "manifest": manifest,
        "train_count": counts["train"],
        "eval_count": counts["validation"],
        "actual_selected_rows": selected,
        "actual_selected_rows_truncated": {
            split: counts[split] > len(selected[split]) for split in selected
        },
        "supervised_tokens": supervised_tokens,
        "data_sha256": _sha256_files(
            (data_dir / "training.jsonl", data_dir / "validation.jsonl")
        ),
        "loading": "disk-backed Arrow cache; batches are paged from storage",
    }
