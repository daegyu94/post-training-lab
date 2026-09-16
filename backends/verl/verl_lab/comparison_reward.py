"""Numeric reward matching NeMo RL's math smoke semantics."""

from __future__ import annotations

import re
from typing import Any


INTEGER_RE = re.compile(r"(?<![\w.])-?\d+(?!\w|\.\d)")


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
) -> float:
    del data_source, extra_info
    matches = INTEGER_RE.findall(solution_str)
    try:
        return float(bool(matches) and int(matches[-1]) == int(ground_truth))
    except (TypeError, ValueError):
        return 0.0
