"""UltraChat subset validation and materialization helpers."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


DATASET_NAME = "HuggingFaceH4/ultrachat_200k"
TRAIN_SPLIT = "train_sft"
EVAL_SPLIT = "test_sft"


def validate_conversation(row: dict[str, Any]) -> bool:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return False
    roles = [message.get("role") for message in messages]
    if roles[0] not in {"system", "user"} or "assistant" not in roles:
        return False
    return all(
        isinstance(message.get("content"), str) and bool(message["content"].strip())
        for message in messages
    )


def select_conversations(
    rows: Iterable[dict[str, Any]], count: int
) -> list[dict[str, Any]]:
    if count < 1:
        raise ValueError("sample count must be positive")
    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row in rows:
        prompt_id = str(row.get("prompt_id", ""))
        if not prompt_id or prompt_id in seen_ids or not validate_conversation(row):
            continue
        selected.append({"prompt_id": prompt_id, "messages": row["messages"]})
        seen_ids.add(prompt_id)
        if len(selected) == count:
            return selected
    raise ValueError(f"requested {count} valid conversations, found {len(selected)}")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary_path.replace(path)
