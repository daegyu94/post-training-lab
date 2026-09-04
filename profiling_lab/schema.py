"""Load and validate the canonical metric schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {"name", "category", "unit", "scope", "source", "policy"}


def load_schema(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise ValueError("Unsupported schema_version")
    metrics = value.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        raise ValueError("metrics must be a non-empty list")
    names: set[str] = set()
    for metric in metrics:
        missing = REQUIRED_FIELDS - metric.keys()
        if missing:
            raise ValueError(f"Metric is missing fields: {sorted(missing)}")
        if metric["name"] in names:
            raise ValueError(f"Duplicate metric name: {metric['name']}")
        names.add(metric["name"])
    return value
