from __future__ import annotations

import json
from pathlib import Path

import pytest

from verl_lab.preflight import ConfigError, load_plan, main


ROOT = Path(__file__).parents[2]
PLAN = ROOT / "experiments" / "verl" / "qwen3-30b-agentic-grpo-smoke.json"


def test_smoke_plan_is_statically_valid(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(PLAN)]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "algorithm": "grpo",
        "model": "Qwen/Qwen3-30B-A3B",
        "rollout_host": "spark@spark1",
        "tool": "calculator",
        "trainer_host": "spark@spark2",
    }


def test_preflight_rejects_single_sample_grpo(tmp_path: Path) -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    plan["rollout"]["samples_per_prompt"] = 1
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ConfigError, match="at least two samples"):
        load_plan(path)


def test_preflight_rejects_long_initial_rollout(tmp_path: Path) -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    plan["rollout"]["max_response_length"] = 513
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ConfigError, match="1024 tokens"):
        load_plan(path)
