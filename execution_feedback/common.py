"""Shared JSONL and schema helpers for execution feedback."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator


VALID_SPLITS = {"train", "validation", "test"}
VALID_STATUSES = {"pass", "fail", "timeout", "infra_error"}


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: each row must be an object")
            yield row


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    temporary.replace(path)
    return count


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_text(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def validate_task(row: dict[str, Any]) -> dict[str, Any]:
    for field in ("task_id", "prompt", "reference_solution", "source", "split"):
        require_text(row, field)
    if row["split"] not in VALID_SPLITS:
        raise ValueError(f"invalid split for {row['task_id']}: {row['split']}")
    tests = row.get("tests")
    if not isinstance(tests, list) or not tests or not all(isinstance(test, str) and test.strip() for test in tests):
        raise ValueError(f"tests must be a non-empty string list for {row['task_id']}")
    setup = row.get("test_setup", "")
    if not isinstance(setup, str):
        raise ValueError(f"test_setup must be a string for {row['task_id']}")
    return row


def index_tasks(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    groups: dict[str, str] = {}
    for row in read_jsonl(path):
        validate_task(row)
        task_id = row["task_id"]
        if task_id in result:
            raise ValueError(f"duplicate task_id: {task_id}")
        group = row.get("template_group")
        if group:
            previous = groups.setdefault(group, row["split"])
            if previous != row["split"]:
                raise ValueError(f"template_group {group!r} crosses splits")
        result[task_id] = row
    return result


def sft_row(task: dict[str, Any], solution: str | None = None) -> dict[str, Any]:
    return {
        "prompt_id": task["task_id"],
        "messages": [
            {"role": "user", "content": task["prompt"]},
            {"role": "assistant", "content": solution or task["reference_solution"]},
        ],
        "provenance": {
            "dataset": task["source"],
            "source_id": task["task_id"],
            "label_source": "verified_execution" if solution else "reference_solution",
        },
    }

