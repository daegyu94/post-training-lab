from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

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


def test_execution_launcher_passes_string_env_and_adapter_rank(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in {
        "hostname": "printf 'spark2\\n'\n",
        "docker": "exit 0\n",
        "python": (
            "if [[ $1 == -c && $2 == *json.load* ]]; then printf '8\\n'; exit; fi\n"
            "printf '<%s>\\n' \"$@\"\n"
        ),
    }.items():
        path = bin_dir / name
        path.write_text(f"#!/usr/bin/env bash\n{body}", encoding="utf-8")
        path.chmod(0o755)

    model_dir = tmp_path / "model"
    adapter_dir = tmp_path / "adapter"
    model_dir.mkdir()
    adapter_dir.mkdir()
    (model_dir / "config.json").write_text("{}\n", encoding="utf-8")
    (adapter_dir / "adapter_config.json").write_text('{"r": 8}\n', encoding="utf-8")
    (adapter_dir / "adapter_model.safetensors").touch()
    tasks = tmp_path / "tasks.jsonl"
    tasks.touch()

    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["bash", str(repo_root / "backends/verl/scripts/run_execution_grpo_smoke.sh")],
        cwd=repo_root,
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "VERL_PYTHON": str(bin_dir / "python"),
            "MODEL_DIR": str(model_dir),
            "LORA_ADAPTER": str(adapter_dir),
            "TASKS": str(tasks),
            "OUTPUT_DIR": str(tmp_path / "output"),
        },
        check=True,
        capture_output=True,
        text=True,
    )

    assert '<+ray_kwargs.ray_init.runtime_env.env_vars.EXECUTION_REWARD_TIMEOUT_SECONDS="10">' in result.stdout
    assert '<+ray_kwargs.ray_init.runtime_env.env_vars.EXECUTION_REWARD_CPUS="1">' in result.stdout
    assert '<+ray_kwargs.ray_init.runtime_env.env_vars.EXECUTION_REWARD_PIDS_LIMIT="64">' in result.stdout
    assert "<+actor_rollout_ref.model.override_config.attn_implementation=sdpa>" in result.stdout
    assert "<actor_rollout_ref.model.lora_rank=8>" in result.stdout
    assert (tmp_path / "output/logs/trainer.log").read_text() in result.stdout
