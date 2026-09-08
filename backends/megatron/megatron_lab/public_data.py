"""Pinned public-dataset adapters for canonical conversational JSONL."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Iterator


ADAPTER_VERSION = "public-sft-v1"
SHA_RE = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class PublicDatasetSpec:
    key: str
    dataset_id: str
    config: str | None
    source_split: str
    license: str
    source_schema: str
    default_revision: str | None = None
    gated: bool = False
    reference_only: bool = False


PRESETS: dict[str, PublicDatasetSpec] = {
    "ultrachat": PublicDatasetSpec(
        "ultrachat", "HuggingFaceH4/ultrachat_200k", "default", "train_sft", "mit", "messages", "8049631c405ae6576f93f445c6b8166f76f5505a"
    ),
    "no_robots": PublicDatasetSpec(
        "no_robots", "HuggingFaceH4/no_robots", "default", "train", "cc-by-nc-4.0", "messages", "e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b"
    ),
    "self_oss": PublicDatasetSpec(
        "self_oss", "bigcode/self-oss-instruct-sc2-exec-filter-50k", "default", "train", "odc-by", "instruction,response", "356bb069eee815daa6e23e9a282eeefe1490ad44"
    ),
    "xlam": PublicDatasetSpec(
        "xlam", "Salesforce/xlam-function-calling-60k", "dataset", "train", "cc-by-4.0", "query,tools,answers (JSON strings)", "26d14ebfe18b1f7b524bd39b404b50af5dc97866", gated=True
    ),
    "swe_trajectories": PublicDatasetSpec(
        "swe_trajectories", "nebius/SWE-agent-trajectories", "default", "train", "cc-by-4.0 plus per-repository terms", "instance_id,model_name,target,trajectory,exit_status,generated_patch,eval_logs", reference_only=True
    ),
}


def get_preset(name: str) -> PublicDatasetSpec:
    try:
        return PRESETS[name]
    except KeyError as exc:
        raise ValueError(f"unknown public dataset preset: {name}") from exc


def resolve_revision(spec: PublicDatasetSpec, revision: str | None) -> str:
    requested = revision or spec.default_revision
    if not requested:
        raise ValueError(f"a pinned revision is required for {spec.dataset_id}")
    if SHA_RE.fullmatch(requested):
        return requested
    try:
        from huggingface_hub import HfApi

        resolved = HfApi().dataset_info(spec.dataset_id, revision=requested).sha
    except Exception as exc:
        raise ValueError(f"could not resolve dataset revision {requested!r}") from exc
    if not resolved or not SHA_RE.fullmatch(resolved):
        raise ValueError(f"Hub returned a non-immutable revision for {spec.dataset_id}")
    return resolved


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _canonical_json(value: Any, field: str) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field} is not valid JSON") from exc
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} cannot be canonicalized") from exc


def _prompt_id(row: dict[str, Any], fallback: str) -> str:
    value = row.get("prompt_id", row.get("id", fallback))
    if value is None:
        raise ValueError("prompt_id is required")
    return _text(str(value), "prompt_id")


def _native_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("messages must contain at least two turns")
    result: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("each message must be an object")
        if "tool_calls" in message or "function_call" in message:
            raise ValueError("structured tool calls are not supported by this adapter")
        role = _text(message.get("role"), "message.role")
        content = _text(message.get("content"), "message.content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported native role: {role}")
        result.append({"role": role, "content": content})
    if result[0]["role"] not in {"system", "user"} or "user" not in {message["role"] for message in result}:
        raise ValueError("native conversations must start with system/user and contain user")
    if result[-1]["role"] != "assistant":
        raise ValueError("canonical SFT examples must end with assistant")
    return result


def adapt_row(spec: PublicDatasetSpec, row: dict[str, Any], ordinal: int = 0) -> dict[str, Any]:
    """Convert one source row; malformed rows fail rather than disappear."""

    if spec.reference_only:
        raise ValueError(f"{spec.key} is reference-only and has no SFT adapter")
    prompt_id = _prompt_id(row, f"row-{ordinal}")
    if spec.key in {"ultrachat", "no_robots"}:
        messages = _native_messages(row)
    elif spec.key == "self_oss":
        messages = [
            {"role": "user", "content": _text(row.get("instruction"), "instruction")},
            {"role": "assistant", "content": _text(row.get("response"), "response")},
        ]
    elif spec.key == "xlam":
        tools = json.loads(_canonical_json(row.get("tools"), "tools"))
        answers = json.loads(_canonical_json(row.get("answers"), "answers"))
        if not isinstance(tools, list) or not isinstance(answers, list):
            raise ValueError("xlam tools and answers must be JSON arrays")
        if not tools or not answers:
            raise ValueError("xlam tools and answers must be non-empty arrays")
        for tool in tools:
            if not isinstance(tool, dict) or not isinstance(tool.get("name"), str) or not isinstance(tool.get("parameters"), dict):
                raise ValueError("each xlam tool needs name and parameters object")
        for answer in answers:
            if not isinstance(answer, dict) or not isinstance(answer.get("name"), str) or not isinstance(answer.get("arguments"), dict):
                raise ValueError("each xlam answer needs name and arguments object")
        tools_text = _canonical_json(tools, "tools")
        answers_text = _canonical_json(answers, "answers")
        messages = [
            {
                "role": "system",
                "content": "Available tools (JSON): " + tools_text + "\nReturn the selected call(s) as JSON.",
            },
            {"role": "user", "content": _text(row.get("query"), "query")},
            {"role": "assistant", "content": answers_text},
        ]
    else:
        raise ValueError(f"no adapter for {spec.key}")
    return {
        "prompt_id": prompt_id,
        "messages": messages,
        "provenance": {
            "dataset": spec.dataset_id,
            "preset": spec.key,
            "source_id": prompt_id,
            "adapter_version": ADAPTER_VERSION,
        },
    }


def _prompt_group(row: dict[str, Any]) -> str:
    """Group duplicate prompts even when source IDs differ."""

    inputs = [message for message in row["messages"] if message["role"] != "assistant"]
    encoded = json.dumps(inputs, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def select_bounded(
    rows: Iterable[dict[str, Any]], spec: PublicDatasetSpec, train_count: int, eval_count: int, seed: int, max_scan: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if train_count < 1 or eval_count < 1 or max_scan < 1:
        raise ValueError("train-count, eval-count, and max-scan must be positive")
    train: list[dict[str, Any]] = []
    evaluation: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_groups: set[str] = set()
    for ordinal, source_row in enumerate(rows):
        if ordinal >= max_scan:
            break
        try:
            row = adapt_row(spec, source_row, ordinal)
        except ValueError:
            raise
        key = row["prompt_id"]
        group = _prompt_group(row)
        if key in seen_ids or group in seen_groups:
            continue
        seen_ids.add(key)
        seen_groups.add(group)
        bucket = int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:8], 16) / 0x100000000
        if bucket < 0.2 and len(evaluation) < eval_count:
            evaluation.append(row)
        elif bucket >= 0.2 and len(train) < train_count:
            train.append(row)
        if len(train) == train_count and len(evaluation) == eval_count:
            return train, evaluation
    raise ValueError(f"bounded stream ended before train={train_count}, validation={eval_count}")


def iter_hub_rows(spec: PublicDatasetSpec, revision: str) -> Iterator[dict[str, Any]]:
    if spec.reference_only:
        raise ValueError(f"{spec.key} is reference-only")
    from datasets import load_dataset

    kwargs: dict[str, Any] = {"split": spec.source_split, "revision": revision, "streaming": True}
    if spec.config is None:
        dataset = load_dataset(spec.dataset_id, **kwargs)
    else:
        dataset = load_dataset(spec.dataset_id, spec.config, **kwargs)
    yield from dataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_public_jsonl(output_dir: Path, spec: PublicDatasetSpec, revision: str, train: list[dict[str, Any]], evaluation: list[dict[str, Any]], seed: int, max_scan: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {"training": output_dir / "training.jsonl", "validation": output_dir / "validation.jsonl"}
    for name, rows in (("training", train), ("validation", evaluation)):
        temporary = paths[name].with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as destination:
            for row in rows:
                destination.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        temporary.replace(paths[name])
    manifest: dict[str, Any] = {
        "manifest_version": 1,
        "adapter_version": ADAPTER_VERSION,
        "preset": spec.key,
        "dataset": spec.dataset_id,
        "config": spec.config,
        "dataset_revision": revision,
        "source_split": spec.source_split,
        "validation_policy": "deterministic holdout from source train split; source test split not read",
        "source_schema": spec.source_schema,
        "license": spec.license,
        "seed": seed,
        "max_scan": max_scan,
        "train_count": len(train),
        "eval_count": len(evaluation),
        "train_prompt_ids": [row["prompt_id"] for row in train],
        "eval_prompt_ids": [row["prompt_id"] for row in evaluation],
        "source_selection_sha256": hashlib.sha256(json.dumps(
            {"train": [row["prompt_id"] for row in train], "validation": [row["prompt_id"] for row in evaluation]},
            sort_keys=True, separators=(",", ":"),
        ).encode()).hexdigest(),
        "files": {name: {"path": path.name, "sha256": _sha256(path)} for name, path in paths.items()},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest
