"""Prepare synthetic or MBPP Python function tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from execution_feedback.common import index_tasks, sft_row, write_jsonl, sha256


SYNTHETIC_TASKS: tuple[dict[str, Any], ...] = (
    {"task_id": "syn-list-001", "prompt": "Implement `unique_sorted(values)` returning sorted unique integers. Return only Python code.", "reference_solution": "def unique_sorted(values):\n    return sorted(set(values))\n", "tests": ["assert unique_sorted([3, 1, 3, 2]) == [1, 2, 3]", "assert unique_sorted([]) == []"], "split": "train", "template_group": "list-set-a"},
    {"task_id": "syn-list-002", "prompt": "Implement `dedupe_ordered(values)` preserving the first occurrence. Return only Python code.", "reference_solution": "def dedupe_ordered(values):\n    return list(dict.fromkeys(values))\n", "tests": ["assert dedupe_ordered([2, 1, 2, 3, 1]) == [2, 1, 3]", "assert dedupe_ordered([]) == []"], "split": "train", "template_group": "list-order-a"},
    {"task_id": "syn-string-001", "prompt": "Implement `word_count(text)` using whitespace-separated words. Return only Python code.", "reference_solution": "def word_count(text):\n    return len(text.split())\n", "tests": ["assert word_count('a  b c') == 3", "assert word_count('') == 0"], "split": "train", "template_group": "string-count-a"},
    {"task_id": "syn-dict-001", "prompt": "Implement `invert_unique(mapping)` for a dictionary whose values are unique. Return only Python code.", "reference_solution": "def invert_unique(mapping):\n    return {value: key for key, value in mapping.items()}\n", "tests": ["assert invert_unique({'a': 1, 'b': 2}) == {1: 'a', 2: 'b'}", "assert invert_unique({}) == {}"], "split": "train", "template_group": "dict-invert-a"},
    {"task_id": "syn-number-001", "prompt": "Implement `clamp(value, low, high)`. Return only Python code.", "reference_solution": "def clamp(value, low, high):\n    return max(low, min(value, high))\n", "tests": ["assert clamp(5, 0, 3) == 3", "assert clamp(-1, 0, 3) == 0", "assert clamp(2, 0, 3) == 2"], "split": "validation", "template_group": "number-bound-a"},
    {"task_id": "syn-string-002", "prompt": "Implement `is_palindrome(text)` ignoring case and non-alphanumeric characters. Return only Python code.", "reference_solution": "def is_palindrome(text):\n    normalized = ''.join(c.lower() for c in text if c.isalnum())\n    return normalized == normalized[::-1]\n", "tests": ["assert is_palindrome('A man, a plan, a canal: Panama')", "assert not is_palindrome('python')"], "split": "validation", "template_group": "string-normalize-a"},
    {"task_id": "syn-list-003", "prompt": "Implement `chunked(values, size)` returning consecutive lists and raising ValueError when size is not positive. Return only Python code.", "reference_solution": "def chunked(values, size):\n    if size <= 0:\n        raise ValueError('size must be positive')\n    return [values[i:i + size] for i in range(0, len(values), size)]\n", "tests": ["assert chunked([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]", "\ntry:\n    chunked([1], 0)\n    assert False\nexcept ValueError:\n    pass"], "split": "test", "template_group": "list-chunk-a"},
    {"task_id": "syn-dict-002", "prompt": "Implement `merge_counts(left, right)` summing matching integer values without mutating inputs. Return only Python code.", "reference_solution": "def merge_counts(left, right):\n    result = dict(left)\n    for key, value in right.items():\n        result[key] = result.get(key, 0) + value\n    return result\n", "tests": ["assert merge_counts({'a': 2}, {'a': 3, 'b': 1}) == {'a': 5, 'b': 1}", "left = {'a': 1}; merge_counts(left, {'a': 1}); assert left == {'a': 1}"], "split": "test", "template_group": "dict-merge-a"},
)


def adapt_mbpp(row: dict[str, Any], split: str) -> dict[str, Any]:
    task_id = str(row["task_id"])
    tests = row.get("test_list")
    if not isinstance(tests, list) or not tests:
        raise ValueError(f"MBPP task {task_id} has no test_list")
    return {
        "task_id": f"mbpp-{task_id}",
        "prompt": f"{row['text'].strip()}\nReturn only Python code.",
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
    parser.add_argument("--source", choices=("synthetic", "mbpp"), default="synthetic")
    parser.add_argument("--revision", help="Required immutable Hub revision for MBPP")
    parser.add_argument("--limit-per-split", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.source == "mbpp":
        if not args.revision or len(args.revision) != 40:
            parser.error("--revision must be a 40-character immutable commit for MBPP")
        rows = load_mbpp(args.revision, args.limit_per_split)
        revision = args.revision
        source = "google-research-datasets/mbpp"
        license_name = "cc-by-4.0"
    else:
        rows = SYNTHETIC_TASKS
        revision = "synthetic-v1"
        source = "synthetic-python-functions"
        license_name = "project-terms"
    manifest = prepare(rows, args.output_dir, source, revision, license_name)
    print(json.dumps(manifest["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
