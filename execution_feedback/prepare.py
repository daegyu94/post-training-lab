"""Prepare pinned MBPP Python function tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any, Iterable

from execution_feedback.common import index_tasks, sft_row, write_jsonl, sha256


MBPP_REVISION = "4bb6404fdc6cacfda99d4ac4205087b89d32030c"


def adapt_mbpp(row: dict[str, Any], split: str) -> dict[str, Any]:
    task_id = str(row["task_id"])
    tests = row.get("test_list")
    if not isinstance(tests, list) or not tests:
        raise ValueError(f"MBPP task {task_id} has no test_list")
    return {
        "task_id": f"mbpp-{task_id}",
        "prompt": f"{row['prompt'].strip()}\nReturn only Python code.",  # MBPP's own 'prompt' field, not this dict's

        "reference_solution": row["code"],
        "test_setup": row.get("test_setup_code", ""),
        "tests": tests,
        "split": split,
        "source": "google-research-datasets/mbpp",
        "template_group": f"mbpp-{task_id}",
    }


def load_mbpp(revision: str, limit_per_split: int | None) -> Iterable[dict[str, Any]]:
    from datasets import load_dataset

    for split in ("train", "validation", "test"):
        dataset = load_dataset("google-research-datasets/mbpp", "sanitized", split=split, revision=revision)
        for ordinal, row in enumerate(dataset):
            if limit_per_split is not None and ordinal >= limit_per_split:
                break
            yield adapt_mbpp(dict(row), split)


def prepare(rows: Iterable[dict[str, Any]], output_dir: Path, source: str, revision: str, license_name: str = "project-terms") -> dict[str, Any]:
    normalized = []
    for row in rows:
        item = dict(row)
        item.setdefault("test_setup", "")
        item.setdefault("source", source)
        normalized.append(item)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_path = output_dir / "tasks.jsonl"
    write_jsonl(all_path, normalized)
    indexed = index_tasks(all_path)
    files: dict[str, dict[str, Any]] = {"tasks": {"path": all_path.name, "sha256": sha256(all_path)}}
    counts: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        selected = [task for task in indexed.values() if task["split"] == split]
        split_path = output_dir / f"{split}.jsonl"
        counts[split] = write_jsonl(split_path, selected)
        files[split] = {"path": split_path.name, "sha256": sha256(split_path)}
        if split in {"train", "validation"}:
            sft_path = output_dir / f"initial_sft_{split}.jsonl"
            write_jsonl(sft_path, (sft_row(task) for task in selected))
            files[f"initial_sft_{split}"] = {"path": sft_path.name, "sha256": sha256(sft_path)}
    manifest = {"manifest_version": 1, "dataset_kind": "execution_feedback", "source": source, "source_revision": revision, "license": license_name, "counts": counts, "files": files}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", default=MBPP_REVISION, help="Immutable MBPP Hub revision")
    parser.add_argument("--limit-per-split", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("--revision must be a 40-character lowercase commit SHA")
    if args.limit_per_split is not None and args.limit_per_split < 1:
        parser.error("--limit-per-split must be positive")
    manifest = prepare(
        load_mbpp(args.revision, args.limit_per_split), args.output_dir,
        "google-research-datasets/mbpp", args.revision, "cc-by-4.0",
    )
    print(json.dumps(manifest["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
