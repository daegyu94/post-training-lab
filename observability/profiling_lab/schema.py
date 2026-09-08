"""Load and validate the canonical metric contract."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {"name", "category", "unit", "scope", "source", "policy"}
ALLOWED_CATEGORIES = {
    "training",
    "rollout",
    "agent",
    "orchestration",
    "gpu",
    "host",
    "container",
    "network",
    "data_movement",
    "storage",
    "checkpoint",
}
SNAKE_CASE = re.compile(r"^[a-z][a-z0-9_]*$")


def _validate_vocabulary(name: str, value: Any) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    if len(value) != len(set(value)):
        raise ValueError(f"{name} must contain unique values")
    if not all(isinstance(item, str) and SNAKE_CASE.fullmatch(item) for item in value):
        raise ValueError(f"{name} must contain snake_case strings")


def load_schema(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("$schema") != "./metrics.schema.json":
        raise ValueError("$schema must reference ./metrics.schema.json")
    if value.get("schema_version") != 1:
        raise ValueError("Unsupported schema_version")

    _validate_vocabulary("recommended_labels", value.get("recommended_labels"))
    _validate_vocabulary("manifest_only_fields", value.get("manifest_only_fields"))
    _validate_vocabulary("phase_vocabulary", value.get("phase_vocabulary"))

    metrics = value.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        raise ValueError("metrics must be a non-empty list")

    names: set[str] = set()
    for metric in metrics:
        if not isinstance(metric, dict):
            raise ValueError("Each metric must be an object")
        missing = REQUIRED_FIELDS - metric.keys()
        if missing:
            raise ValueError(f"Metric is missing fields: {sorted(missing)}")
        extra = metric.keys() - REQUIRED_FIELDS
        if extra:
            raise ValueError(f"Metric has unsupported fields: {sorted(extra)}")
        if not isinstance(metric["name"], str) or not SNAKE_CASE.fullmatch(metric["name"]):
            raise ValueError(f"Invalid metric name: {metric['name']!r}")
        if metric["category"] not in ALLOWED_CATEGORIES:
            raise ValueError(f"Unsupported metric category: {metric['category']}")
        for field in REQUIRED_FIELDS - {"name", "category"}:
            if not isinstance(metric[field], str) or not metric[field].strip():
                raise ValueError(f"Metric field {field!r} must be a non-empty string")
        if metric["name"] in names:
            raise ValueError(f"Duplicate metric name: {metric['name']}")
        names.add(metric["name"])
    return value
