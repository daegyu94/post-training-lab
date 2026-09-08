"""Build the common machine-readable experiment summary."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


SUMMARY_SCHEMA_VERSION = 1
SUMMARY_SECTIONS = (
    "configuration",
    "environment",
    "quality",
    "performance",
    "artifacts",
    "validation",
)


def make_run_summary(
    *,
    configuration: Mapping[str, Any],
    environment: Mapping[str, Any] | None = None,
    quality: Mapping[str, Any] | None = None,
    performance: Mapping[str, Any] | None = None,
    artifacts: Mapping[str, Any] | None = None,
    validation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a versioned summary with stable top-level sections."""
    values = {
        "configuration": configuration,
        "environment": environment,
        "quality": quality,
        "performance": performance,
        "artifacts": artifacts,
        "validation": validation,
    }
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        **{section: dict(values[section] or {}) for section in SUMMARY_SECTIONS},
    }
