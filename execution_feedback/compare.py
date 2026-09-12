"""Compare A/B/C/D test evaluations using strict pass@1."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from execution_feedback.common import read_jsonl


def summarize(path: Path) -> dict[str, Any]:
    rows = list(read_jsonl(path))
    if not rows:
        raise ValueError(f"empty evaluation file: {path}")
    by_task: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("split") != "test":
            raise ValueError(f"comparison accepts test rows only: {row.get('task_id')}")
        if str(row.get("candidate_id", "0")) != "0":
            raise ValueError("pass@1 comparison requires candidate_id=0 only")
        task_id = row.get("task_id")
        if task_id in by_task:
            raise ValueError(f"duplicate test result: {task_id}")
        by_task[task_id] = row
    counts = Counter(row.get("status") for row in rows)
    passed = counts["pass"]
    return {
        "evaluation_file": str(path), "task_ids": sorted(by_task), "task_count": len(rows),
        "pass_at_1": passed / len(rows), "status_counts": {name: counts[name] for name in ("pass", "fail", "timeout", "infra_error")},
        "evaluation_seconds": sum(float(row.get("duration_seconds", 0.0)) for row in rows),
    }


def compare(variants: dict[str, Path]) -> dict[str, Any]:
    summaries = {name: summarize(path) for name, path in variants.items()}
    expected = None
    for name, summary in summaries.items():
        ids = summary.pop("task_ids")
        if expected is None:
            expected = ids
        elif ids != expected:
            raise ValueError(f"variant {name} does not contain the same test task IDs")
    return {"metric": "strict pass@1: one candidate and all tests pass", "test_task_ids": expected, "variants": summaries}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", action="append", required=True, metavar="NAME=EVALUATIONS_JSONL")
    parser.add_argument("--training-summary", action="append", default=[], metavar="NAME=SUMMARY_JSON")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    variants: dict[str, Path] = {}
    for value in args.variant:
        if "=" not in value:
            parser.error("--variant must use NAME=PATH")
        name, path = value.split("=", 1)
        if name in variants:
            parser.error(f"duplicate variant: {name}")
        variants[name] = Path(path)
    result = compare(variants)
    for value in args.training_summary:
        if "=" not in value:
            parser.error("--training-summary must use NAME=PATH")
        name, path = value.split("=", 1)
        if name not in result["variants"]:
            parser.error(f"training summary has no matching variant: {name}")
        summary = json.loads(Path(path).read_text(encoding="utf-8"))
        training = result["variants"][name].setdefault("additional_training", {"stages": [], "total_duration_seconds": 0.0, "total_input_tokens_seen": 0})
        stage = {
            "duration_seconds": summary.get("duration_seconds"),
            "num_input_tokens_seen": summary.get("num_input_tokens_seen"),
            "max_steps": summary.get("max_steps"),
            "mode": summary.get("mode"),
            "metrics": summary.get("metrics", {}),
        }
        training["stages"].append(stage)
        training["total_duration_seconds"] += float(stage["duration_seconds"] or 0.0)
        training["total_input_tokens_seen"] += int(stage["num_input_tokens_seen"] or 0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({name: summary["pass_at_1"] for name, summary in result["variants"].items()}, sort_keys=True))


if __name__ == "__main__":
    main()
