from __future__ import annotations

import json

import pytest

from verl_lab.execution_reward import RewardInfrastructureError, compute_score
from verl_lab.prepare_execution import reward_payload, row, rows_by_split


def task(split: str = "train") -> dict[str, object]:
    return {
        "task_id": "mbpp-1", "prompt": "Write add.", "reference_solution": "def add(a, b): return a + b",
        "source": "mbpp", "split": split, "test_setup": "", "tests": ["assert add(1, 2) == 3"],
    }


def test_execution_row_excludes_reference_solution() -> None:
    item = row(task())
    encoded = json.dumps(item)
    assert "def add(a, b): return a + b" not in encoded
    assert item["extra_info"] == {"task_id": "mbpp-1", "split": "train"}
    assert json.loads(item["reward_model"]["ground_truth"])["tests"] == ["assert add(1, 2) == 3"]


def test_reward_payload_keeps_only_grading_fields() -> None:
    payload = json.loads(reward_payload(task()))
    assert set(payload) == {"source", "split", "task_id", "test_setup", "tests"}


def test_execution_reward_maps_pass_fail_and_timeout(monkeypatch) -> None:
    payload = reward_payload(task())
    monkeypatch.setattr(
        "verl_lab.execution_reward.evaluate_candidate",
        lambda *args, **kwargs: {"status": "pass"},
    )
    assert compute_score("execution_feedback", "def add(a, b): return a + b", payload) == 1.0
    monkeypatch.setattr(
        "verl_lab.execution_reward.evaluate_candidate",
        lambda *args, **kwargs: {"status": "fail"},
    )
    assert compute_score("execution_feedback", "def add(a, b): return 0", payload) == 0.0
    monkeypatch.setattr(
        "verl_lab.execution_reward.evaluate_candidate",
        lambda *args, **kwargs: {"status": "timeout"},
    )
    assert compute_score("execution_feedback", "while True: pass", payload) == 0.0


def test_execution_reward_aborts_on_infrastructure_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        "verl_lab.execution_reward.evaluate_candidate",
        lambda *args, **kwargs: {"status": "infra_error", "failures": ["docker unavailable"]},
    )
    with pytest.raises(RewardInfrastructureError, match="docker unavailable"):
        compute_score("execution_feedback", "pass", reward_payload(task()))


def test_rows_require_train_and_validation(tmp_path) -> None:
    path = tmp_path / "tasks.jsonl"
    path.write_text(json.dumps(task()) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="validation"):
        rows_by_split(path)
