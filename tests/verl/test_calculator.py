from __future__ import annotations

import pytest

from verl_lab.calculator import ToolError, calculate, score_answer
from verl_lab.reward import compute_score


def test_calculator_allows_only_small_integer_arithmetic() -> None:
    assert calculate("2 * (7 - 4) + 1") == 7
    assert calculate("-3 * 4") == -12


@pytest.mark.parametrize("expression", ("2 ** 20", "1 / 2", "__import__('os')", "True + 1"))
def test_calculator_rejects_non_smoke_task_syntax(expression: str) -> None:
    with pytest.raises(ToolError):
        calculate(expression)


def test_score_answer_requires_exact_final_value() -> None:
    assert score_answer("7", 7) == 1.0
    assert score_answer(" 7\n", 7) == 1.0
    assert score_answer("The answer is 7", 7) == 0.0


def test_rule_reward_reads_the_last_marked_final_answer() -> None:
    assert compute_score("calculator", "tool said 9; #### 9", "9") == 1.0
    assert compute_score("calculator", "#### 8", "9") == 0.0
