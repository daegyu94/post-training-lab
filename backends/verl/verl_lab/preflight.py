"""Validate a role-separated Verl agentic-RL plan without contacting Spark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any


REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


class ConfigError(ValueError):
    """Raised when the static agentic-RL plan is incomplete or unsafe."""


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{label} must be an object")
    return value


def _keys(value: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ConfigError(f"{label} has unsupported fields: {', '.join(unknown)}")


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{label} must be a non-empty string")
    return value


def _positive(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ConfigError(f"{label} must be a positive integer")
    return value


def load_plan(path: Path) -> dict[str, Any]:
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read plan {path}: {exc}") from exc
    plan = _object(plan, "plan")
    _keys(plan, {"backend", "model", "roles", "rollout", "training", "agent"}, "plan")
    if plan.get("backend") != "verl":
        raise ConfigError("plan.backend must be 'verl'")

    model = _object(plan.get("model"), "plan.model")
    _keys(model, {"id", "revision"}, "plan.model")
    _string(model.get("id"), "plan.model.id")
    revision = _string(model.get("revision"), "plan.model.revision")
    if not REVISION_RE.fullmatch(revision):
        raise ConfigError("plan.model.revision must be a 40-hex immutable revision")

    roles = _object(plan.get("roles"), "plan.roles")
    if set(roles) != {"rollout", "trainer"}:
        raise ConfigError("plan.roles must contain exactly rollout and trainer")
    for name in ("rollout", "trainer"):
        role = _object(roles[name], f"plan.roles.{name}")
        _keys(role, {"host", "python", "model_dir", "output_dir"}, f"plan.roles.{name}")
        for field in ("host", "python", "model_dir", "output_dir"):
            _string(role.get(field), f"plan.roles.{name}.{field}")
    if roles["rollout"]["host"] == roles["trainer"]["host"]:
        raise ConfigError("rollout and trainer must use different hosts")

    rollout = _object(plan.get("rollout"), "plan.rollout")
    _keys(rollout, {"backend", "mode", "max_prompt_length", "max_response_length", "samples_per_prompt"}, "plan.rollout")
    if rollout.get("backend") != "vllm" or rollout.get("mode") != "async":
        raise ConfigError("plan.rollout requires backend=vllm and mode=async")
    for field in ("max_prompt_length", "max_response_length", "samples_per_prompt"):
        _positive(rollout.get(field), f"plan.rollout.{field}")
    if rollout["max_prompt_length"] + rollout["max_response_length"] > 1024:
        raise ConfigError("initial smoke limits prompt plus response to 1024 tokens")
    if rollout["samples_per_prompt"] < 2:
        raise ConfigError("GRPO requires at least two samples per prompt")

    training = _object(plan.get("training"), "plan.training")
    _keys(training, {"algorithm", "finetuning", "lora_rank", "max_steps"}, "plan.training")
    if training.get("algorithm") != "grpo" or training.get("finetuning") != "lora":
        raise ConfigError("plan.training requires GRPO with LoRA")
    for field in ("lora_rank", "max_steps"):
        _positive(training.get(field), f"plan.training.{field}")

    agent = _object(plan.get("agent"), "plan.agent")
    _keys(agent, {"tool", "max_turns"}, "plan.agent")
    if agent.get("tool") != "calculator":
        raise ConfigError("plan.agent.tool must be calculator for the initial smoke task")
    _positive(agent.get("max_turns"), "plan.agent.max_turns")
    return plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    args = parser.parse_args(argv)
    plan = load_plan(args.plan)
    summary = {
        "model": plan["model"]["id"],
        "rollout_host": plan["roles"]["rollout"]["host"],
        "trainer_host": plan["roles"]["trainer"]["host"],
        "algorithm": plan["training"]["algorithm"],
        "tool": plan["agent"]["tool"],
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
