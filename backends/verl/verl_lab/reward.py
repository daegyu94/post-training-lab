"""Rule reward for the calculator agentic-RL smoke task."""

from __future__ import annotations

import re
from typing import Any

from verl_lab.calculator import score_answer


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
    return score_answer(matches[-1] if matches else "", expected)
