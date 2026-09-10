"""Convert reviewed LLM service traces into SFT and evaluation JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ALLOWED_ROLES = {"system", "user", "assistant"}
ALLOWED_LABEL_SOURCES = {"expert", "human_reviewed", "verified_execution"}
SPLITS = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    return parser.parse_args()


def _qualification_error(row: dict[str, Any]) -> str | None:
    review = row.get("review")
    if not isinstance(review, dict) or review.get("status") != "approved":
        return "not_approved"
    if review.get("sensitive_data_removed") is not True:
        return "sensitive_data_not_cleared"
    if review.get("label_source") not in ALLOWED_LABEL_SOURCES:
        return "untrusted_label_source"
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return "invalid_messages"
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in ALLOWED_ROLES:
            return "unsupported_role"
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            return "empty_content"
    if messages[-1]["role"] != "assistant":
        return "missing_reference_answer"
    return None


def _hash_fraction(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _choose_split(
    row: dict[str, Any], seed: int, train_ratio: float, validation_ratio: float
) -> str:
    explicit = row.get("split")
    if explicit is not None:
        if explicit not in SPLITS:
            raise ValueError(f"invalid split: {explicit}")
        return str(explicit)
    value = _hash_fraction(str(row["session_id"]), seed)
    if value < train_ratio:
        return "train"
    if value < train_ratio + validation_ratio:
        return "validation"
    return "test"


def _prompt_fingerprint(messages: list[dict[str, str]]) -> str:
    canonical = json.dumps(messages[:-1], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_dataset(
    input_path: Path,
    output_dir: Path,
    seed: int = 42,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
) -> dict[str, Any]:
    if not 0 < train_ratio < 1 or not 0 <= validation_ratio < 1:
        raise ValueError("split ratios are out of range")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("train-ratio + validation-ratio must be less than 1")

    source_bytes = input_path.read_bytes()
    records = [
        json.loads(line)
        for line in source_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]
    output: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    exclusions: Counter[str] = Counter()
    seen_trace_ids: set[str] = set()
    seen_prompts: set[str] = set()
    session_splits: dict[str, str] = {}

    for row in records:
        trace_id = str(row.get("trace_id", "")).strip()
        session_id = str(row.get("session_id", "")).strip()
        if not trace_id or not session_id:
            exclusions["missing_identity"] += 1
            continue
        if trace_id in seen_trace_ids:
            raise ValueError(f"duplicate trace_id: {trace_id}")
        seen_trace_ids.add(trace_id)
        reason = _qualification_error(row)
        if reason:
            exclusions[reason] += 1
            continue
        messages = row["messages"]
        fingerprint = _prompt_fingerprint(messages)
        if fingerprint in seen_prompts:
            exclusions["duplicate_prompt"] += 1
            continue
        seen_prompts.add(fingerprint)
        split = _choose_split(row, seed, train_ratio, validation_ratio)
        previous_split = session_splits.setdefault(session_id, split)
        if previous_split != split:
            raise ValueError(f"session {session_id} spans {previous_split} and {split}")
        if split == "test":
            output[split].append(
                {
                    "prompt_id": trace_id,
                    "prompt": messages[:-1],
                    "reference_answer": messages[-1]["content"],
                    "grader": row.get("evaluation", {"type": "human_rubric"}),
                }
            )
        else:
            output[split].append({"prompt_id": trace_id, "messages": messages})

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "training.jsonl", output["train"])
    _write_jsonl(output_dir / "validation.jsonl", output["validation"])
    _write_jsonl(output_dir / "test.jsonl", output["test"])
    manifest = {
        "schema_version": 1,
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "split_unit": "session_id",
        "seed": seed,
        "counts": {split: len(output[split]) for split in SPLITS},
        "excluded": dict(sorted(exclusions.items())),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    args = parse_args()
    manifest = prepare_dataset(
        args.input,
        args.output_dir,
        args.seed,
        args.train_ratio,
        args.validation_ratio,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
