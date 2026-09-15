"""Stateless tools loaded by Verl's agent loop."""

from verl.tools.function_tool import function_tool

from verl_lab.calculator import ToolError, calculate


@function_tool("calculator")
def calculator(expression: str) -> str:
    """Evaluate a small integer arithmetic expression.

    Args:
        expression: An expression using integer literals, parentheses, +, -, and *.
    """
    try:
        return str(calculate(expression))
    except ToolError as exc:
        return f"error: {exc}"
