from __future__ import annotations

from pathlib import Path

from verl_lab.comparison_reward import compute_score
from verl_lab.prepare_comparison import rows


ROOT = Path(__file__).parents[2]


def test_comparison_data_preserves_shared_questions_and_answers() -> None:
    data = rows(ROOT / "experiments" / "rl-framework-comparison" / "math-smoke.jsonl")
    assert [row["reward_model"]["ground_truth"] for row in data] == ["9", "42"]
    assert all(row["prompt"][0]["role"] == "user" for row in data)


def test_comparison_reward_uses_the_last_integer() -> None:
    assert compute_score("math-smoke", "The result is 9.", "9") == 1.0
    assert compute_score("math-smoke", "7 - 4 = 3, so #### 9", "9") == 1.0
    assert compute_score("math-smoke", "The result is 8.", "9") == 0.0
