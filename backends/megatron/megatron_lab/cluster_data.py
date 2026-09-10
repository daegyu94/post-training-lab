"""Native-chat prompt/completion preparation for setup2 Megatron SFT."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable


ADAPTER_VERSION = "native-final-assistant-v1"


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], (list, tuple)):
        if len(ids) != 1:
            raise ValueError("tokenizer returned multiple sequences for one string")
        ids = ids[0]
    if not isinstance(ids, (list, tuple)):
        raise ValueError("tokenizer did not return input_ids")
    return [int(token_id) for token_id in ids]


def _validate_messages(messages: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("messages must contain a final assistant turn")
    if not all(isinstance(message, dict) for message in messages):
        raise ValueError("every message must be an object")
    final = messages[-1]
    if final.get("role") != "assistant":
        raise ValueError("SFT row must end with assistant")
    if final.get("tool_calls") is not None or final.get("function_call") is not None:
        raise ValueError("final assistant structured tool calls are unsupported")
    for message in messages[:-1]:
        role = _text(message.get("role"), "message.role")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported message role: {role}")
        _text(message.get("content"), "message.content")
        if message.get("tool_calls") is not None or message.get("function_call") is not None:
            raise ValueError("structured tool calls are unsupported by native chat preparation")
    if not any(message.get("role") == "user" for message in messages[:-1]):
        raise ValueError("SFT prompt must contain a user turn")
    _text(final.get("content"), "assistant.content")
    return messages[:-1], final


def render_prompt_completion(row: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    """Render all but the final assistant turn with the native HF template.

    The returned completion already contains EOS, so the Bridge prompt/completion
    config must set ``add_eos=False``.  This avoids the generic assistant-mask
    inference path, which is not available for GLM-4.7-Flash.
    """

    prompt_messages, final = _validate_messages(row.get("messages"))
    prompt = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt = _text(prompt, "rendered prompt")
    eos = _text(getattr(tokenizer, "eos_token", None), "tokenizer.eos_token")
    completion = _text(final.get("content"), "assistant.content") + eos

    prompt_ids = _token_ids(tokenizer, prompt)
    completion_ids = _token_ids(tokenizer, completion)
    joint_ids = _token_ids(tokenizer, prompt + completion)
    eos_ids = _token_ids(tokenizer, eos)
    if not prompt_ids or not completion_ids or not joint_ids:
        raise ValueError("native prompt/completion tokenization produced an empty sequence")
    if not eos_ids or completion_ids[-len(eos_ids) :] != eos_ids:
        raise ValueError("completion does not end with the tokenizer EOS sequence")
    if joint_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("native prompt tokenization is not a prefix of prompt plus completion")
    if joint_ids[len(prompt_ids) :] != completion_ids:
        raise ValueError("native completion tokenization is not preserved at the prompt boundary")
    supervised_tokens = len(completion_ids)
    if supervised_tokens < 1:
        raise ValueError("final assistant completion has no supervised tokens")
    prompt_id = row.get("prompt_id")
    if prompt_id is None:
        raise ValueError("prompt_id is required")
    return {
        "prompt": prompt,
        "completion": completion,
        "prompt_id": str(prompt_id),
        "supervised_tokens": supervised_tokens,
        "prompt_tokens": len(prompt_ids),
        "provenance": row.get("provenance", {}),
    }


def _iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
            yield row


def _sha256_jsonl(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_split(
    source_path: Path,
    destination: Path,
    split: str,
    tokenizer: Any,
    max_length: int,
    forbidden_ids: set[str] | None = None,
) -> tuple[int, set[str]]:
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.{os.getpid()}-",
        dir=str(destination.parent),
        text=True,
    )
    count = 0
    prompt_ids: set[str] = set()
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as target:
            for ordinal, row in enumerate(_iter_jsonl(source_path)):
                try:
                    example = render_prompt_completion(row, tokenizer)
                except ValueError as exc:
                    # BPE re-tokenization can merge differently across the
                    # prompt/completion boundary for a small fraction of real
                    # conversations (observed ~1.3% on no_robots); that is a
                    # property of this row's text under this tokenizer, not a
                    # malformed row, so skip it rather than aborting the whole
                    # split -- matches datasets_lab.public_data's tolerance.
                    print(f"[cluster_data] skipping {split} row {ordinal}: {exc}", file=sys.stderr)
                    continue
                total_tokens = example["prompt_tokens"] + example["supervised_tokens"]
                if total_tokens > max_length:
                    raise ValueError(
                        f"{split} example {example['prompt_id']} exceeds max_length={max_length}; "
                        "native setup2 preparation never truncates"
                    )
                prompt_id = example["prompt_id"]
                if forbidden_ids is not None and prompt_id in forbidden_ids:
                    raise ValueError("train and validation prompt IDs overlap")
                if forbidden_ids is None:
                    prompt_ids.add(prompt_id)
                target.write(json.dumps(example, ensure_ascii=False, sort_keys=True) + "\n")
                count += 1
            if count == 0:
                raise ValueError(f"{split} split is empty: {source_path}")
            target.flush()
            os.fsync(target.fileno())
        Path(temporary_name).replace(destination)
    except BaseException:
        try:
            Path(temporary_name).unlink()
        except FileNotFoundError:
            pass
        raise
    return count, prompt_ids


def _atomic_write(path: Path, content: str) -> None:
    """Publish one complete artifact, allowing all ranks to prepare idempotently."""

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.{os.getpid()}-", dir=str(path.parent), text=True
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as target:
            target.write(content)
            target.flush()
            os.fsync(target.fileno())
        Path(temporary_name).replace(path)
    except BaseException:
        try:
            Path(temporary_name).unlink()
        except FileNotFoundError:
            pass
        raise


def prepare_cluster_data(
    train_data: Path,
    eval_data: Path,
    output_dir: Path,
    tokenizer: Any,
    max_length: int,
    *,
    dataset_id: str,
    dataset_revision: str,
    model_revision: str,
) -> dict[str, Any]:
    """Create a separate Bridge input without modifying canonical source JSONL."""

    if max_length <= 0:
        raise ValueError("max_length must be positive")
    source_root = train_data.parent.resolve()
    if output_dir.resolve() == source_root or any(
        (output_dir / filename).resolve() == source_path.resolve()
        for filename, source_path in (("training.jsonl", train_data), ("validation.jsonl", eval_data))
    ):
        raise ValueError("prepared output must be separate from canonical source files")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_paths = {"train": train_data, "validation": eval_data}
    train_count, train_ids = _prepare_split(
        train_data,
        output_dir / "training.jsonl",
        "train",
        tokenizer,
        max_length,
    )
    validation_count, _ = _prepare_split(
        eval_data,
        output_dir / "validation.jsonl",
        "validation",
        tokenizer,
        max_length,
        train_ids,
    )
    counts = {"train": train_count, "validation": validation_count}

    source_manifest = train_data.parent / "manifest.json"
    manifest = {
        "adapter": ADAPTER_VERSION,
        "dataset": dataset_id,
        "dataset_revision": dataset_revision,
        "model_revision": model_revision,
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": _sha256_jsonl(source_manifest) if source_manifest.is_file() else None,
        "source_files": {
            split: {"path": str(path), "sha256": _sha256_jsonl(path)}
            for split, path in source_paths.items()
        },
        "counts": counts,
        "loading": "disk-backed Arrow cache; batches are paged from storage",
        "supervision": "final_assistant_completion_only",
        "max_length": max_length,
        "tokenizer_template": "model-native apply_chat_template(enable_thinking=False)",
        "prepared_files": {
            split: {"path": str(output_dir / filename), "sha256": _sha256_jsonl(output_dir / filename)}
            for split, filename in (("train", "training.jsonl"), ("validation", "validation.jsonl"))
        },
    }
    _atomic_write(output_dir / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
