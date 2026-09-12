"""Build execution-filtered SFT examples and DPO pairs from one candidate pool."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

from execution_feedback.common import index_tasks, read_jsonl, sft_row, sha256, write_jsonl


def build_feedback(
    tasks_path: Path, evaluations_path: Path, output_dir: Path,
    max_sft_per_task: int = 1, max_pairs_per_task: int = 1,
) -> dict[str, Any]:
    if max_sft_per_task < 1 or max_pairs_per_task < 1:
        raise ValueError("per-task limits must be positive")
    tasks = index_tasks(tasks_path)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(evaluations_path):
        task_id = row.get("task_id")
        if task_id not in tasks:
            raise ValueError(f"evaluation references unknown task: {task_id}")
        if row.get("split") != tasks[task_id]["split"]:
            raise ValueError(f"split mismatch for {task_id}")
        if tasks[task_id]["split"] != "train":
            raise ValueError(f"feedback data must be train-only: {task_id}")
        grouped[task_id].append(row)

    filtered: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    eligible_tasks = 0
    for task_id in sorted(grouped):
        task = tasks[task_id]
        rows = grouped[task_id]
        passed = sorted((row for row in rows if row.get("status") == "pass"), key=lambda row: str(row.get("candidate_id")))
        failed = sorted((row for row in rows if row.get("status") == "fail"), key=lambda row: (float(row.get("score", 0.0)), str(row.get("candidate_id"))))
        for row in passed[:max_sft_per_task]:
            item = sft_row(task, row["code"])
            item["provenance"].update({"candidate_id": row["candidate_id"], "evaluation_score": row["score"]})
            filtered.append(item)
        if passed and failed:
            eligible_tasks += 1
            chosen_rows = passed[:max_pairs_per_task]
            rejected_rows = failed[:max_pairs_per_task]
            for chosen, rejected in zip(chosen_rows, rejected_rows):
                if chosen["code"].strip() == rejected["code"].strip():
                    continue
                pairs.append({
                    "prompt_id": task_id,
                    "prompt": [{"role": "user", "content": task["prompt"]}],
                    "chosen": [{"role": "assistant", "content": chosen["code"]}],
                    "rejected": [{"role": "assistant", "content": rejected["code"]}],
                    "provenance": {"candidate_pool_sha256": sha256(evaluations_path), "chosen_candidate_id": chosen["candidate_id"], "rejected_candidate_id": rejected["candidate_id"]},
                })

    output_dir.mkdir(parents=True, exist_ok=True)
    filtered_path = output_dir / "filtered_sft.jsonl"
    dpo_path = output_dir / "dpo.jsonl"
    write_jsonl(filtered_path, filtered)
    write_jsonl(dpo_path, pairs)
    manifest = {
        "manifest_version": 1,
        "policy": "train-only; pass->filtered SFT; pass/fail from same pool->DPO; timeout and infra_error excluded",
        "tasks_sha256": sha256(tasks_path), "evaluations_sha256": sha256(evaluations_path),
        "max_sft_per_task": max_sft_per_task, "max_pairs_per_task": max_pairs_per_task,
        "filtered_sft_count": len(filtered), "dpo_pair_count": len(pairs), "dpo_eligible_task_count": eligible_tasks,
        "files": {"filtered_sft": {"path": filtered_path.name, "sha256": sha256(filtered_path)}, "dpo": {"path": dpo_path.name, "sha256": sha256(dpo_path)}},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--evaluations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-sft-per-task", type=int, default=1)
    parser.add_argument("--max-pairs-per-task", type=int, default=1)
    args = parser.parse_args()
    manifest = build_feedback(args.tasks, args.evaluations, args.output_dir, args.max_sft_per_task, args.max_pairs_per_task)
    print(json.dumps({"filtered_sft_count": manifest["filtered_sft_count"], "dpo_pair_count": manifest["dpo_pair_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
