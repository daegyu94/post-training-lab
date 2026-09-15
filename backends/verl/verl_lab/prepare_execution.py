"""Convert execution-feedback tasks into Verl rule-reward datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from execution_feedback.common import index_tasks, sha256


def reward_payload(task: dict[str, Any]) -> str:
    """Serialize exactly the task data needed by the Docker reward."""
    return json.dumps(
        {
            "source": task["source"],
            "split": task["split"],
            "task_id": task["task_id"],
            "test_setup": task.get("test_setup", ""),
            "tests": task["tests"],
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def row(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "data_source": "execution_feedback",
        "prompt": [{"role": "user", "content": task["prompt"]}],
        "reward_model": {"style": "rule", "ground_truth": reward_payload(task)},
        "extra_info": {"task_id": task["task_id"], "split": task["split"]},
    }


def rows_by_split(tasks_path: Path) -> dict[str, list[dict[str, Any]]]:
    result = {"train": [], "validation": []}
    for task in index_tasks(tasks_path).values():
        if task["split"] in result:
            result[task["split"]].append(row(task))
    if not result["train"] or not result["validation"]:
        raise ValueError("execution RLVR requires non-empty train and validation splits")
    return result


def prepare(tasks_path: Path, output_dir: Path) -> dict[str, Any]:
    from datasets import Dataset

    groups = rows_by_split(tasks_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for split, rows in groups.items():
        name = "val" if split == "validation" else split
        path = output_dir / f"{name}.parquet"
        Dataset.from_list(rows).to_parquet(path)
        files[split] = {"path": path.name, "count": len(rows), "sha256": sha256(path)}
    manifest = {
        "manifest_version": 1,
        "kind": "execution_rlvr",
        "tasks_sha256": sha256(tasks_path),
        "reward": "Docker execution: pass=1, fail/timeout=0, infra_error=abort",
        "files": files,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = prepare(args.tasks, args.output_dir)
    print(json.dumps({name: item["count"] for name, item in manifest["files"].items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
