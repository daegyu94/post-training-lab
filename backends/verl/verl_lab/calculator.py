"""Deterministic arithmetic tool and reward used by the agentic-RL smoke task."""

from __future__ import annotations

import ast
import operator


class ToolError(ValueError):
    """Raised when a calculator request is not in the smoke-task language."""


_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
}


def calculate(expression: str) -> int:
    """Evaluate a small integer-only arithmetic expression without ``eval``."""
    if not isinstance(expression, str) or not expression or len(expression) > 128:
        raise ToolError("expression must be a non-empty string of at most 128 characters")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolError("expression is invalid") from exc
    result = _evaluate(tree.body)
    if abs(result) > 1_000_000:
        raise ToolError("result is outside the smoke-task range")
    return result


def _evaluate(node: ast.AST) -> int:
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        left, right = _evaluate(node.left), _evaluate(node.right)
        return _OPERATORS[type(node.op)](left, right)
    raise ToolError("only integer literals and +, -, * are supported")


def score_answer(answer: str, expected: int) -> float:
    """Return the verifiable terminal reward for an exact calculator answer."""
    return float(isinstance(answer, str) and answer.strip() == str(expected))
