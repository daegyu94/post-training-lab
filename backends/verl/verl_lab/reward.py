"""Rule reward for the calculator agentic-RL smoke task."""

from __future__ import annotations

import re
from typing import Any

FINAL_ANSWER_RE = re.compile(r"####\s*(-?\d+)")


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
) -> float:
    """Return one only when the final ``####`` answer matches the ground truth."""
    del data_source, extra_info
    matches = FINAL_ANSWER_RE.findall(solution_str)
    try:
        expected = int(ground_truth)
    except (TypeError, ValueError):
        return 0.0
    return float(bool(matches) and matches[-1].strip() == str(expected))
