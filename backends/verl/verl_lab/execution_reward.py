"""Docker-backed binary reward for code-generation RLVR."""

from __future__ import annotations

import json
import os
from typing import Any

from execution_feedback.evaluate import evaluate_candidate


class RewardInfrastructureError(RuntimeError):
    """Raised when a rollout cannot be graded reliably."""


def _positive_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, default))
    except ValueError as exc:
        raise RewardInfrastructureError(f"{name} must be a positive number") from exc
    if value <= 0:
        raise RewardInfrastructureError(f"{name} must be a positive number")
    return value


def _positive_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, default))
    except ValueError as exc:
        raise RewardInfrastructureError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise RewardInfrastructureError(f"{name} must be a positive integer")
    return value


def _task(ground_truth: str) -> dict[str, Any]:
    try:
        task = json.loads(ground_truth)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RewardInfrastructureError("invalid execution reward payload") from exc
    if not isinstance(task, dict):
        raise RewardInfrastructureError("execution reward payload must be an object")
    for name in ("task_id", "source", "split"):
        if not isinstance(task.get(name), str) or not task[name]:
            raise RewardInfrastructureError(f"execution reward payload lacks {name}")
    if not isinstance(task.get("test_setup", ""), str):
        raise RewardInfrastructureError("execution reward test_setup must be text")
    tests = task.get("tests")
    if not isinstance(tests, list) or not tests or not all(isinstance(test, str) and test for test in tests):
        raise RewardInfrastructureError("execution reward tests must be a non-empty text list")
    return task


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
) -> float:
    """Return a strict execution reward; never turn infrastructure loss into 0."""
    del data_source, extra_info
    task = _task(ground_truth)
    result = evaluate_candidate(
        task,
        {"candidate_id": "rollout", "response": solution_str},
        engine="docker",
        image=os.environ.get("EXECUTION_REWARD_IMAGE", "python:3.12-slim"),
        timeout_seconds=_positive_float("EXECUTION_REWARD_TIMEOUT_SECONDS", 10.0),
        memory=os.environ.get("EXECUTION_REWARD_MEMORY", "256m"),
        cpus=_positive_float("EXECUTION_REWARD_CPUS", 1.0),
        pids_limit=_positive_int("EXECUTION_REWARD_PIDS_LIMIT", 64),
    )
    if result["status"] == "infra_error":
        detail = "; ".join(str(item) for item in result.get("failures", []))
        raise RewardInfrastructureError(f"Docker execution reward unavailable: {detail}")
    return float(result["status"] == "pass")
