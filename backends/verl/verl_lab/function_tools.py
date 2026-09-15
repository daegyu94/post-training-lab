"""Stateless tools loaded by Verl's agent loop."""

import importlib.util
from pathlib import Path

from verl.tools.function_tool import function_tool


_calculator_spec = importlib.util.spec_from_file_location("verl_calculator", Path(__file__).with_name("calculator.py"))
assert _calculator_spec and _calculator_spec.loader
_calculator_module = importlib.util.module_from_spec(_calculator_spec)
_calculator_spec.loader.exec_module(_calculator_module)
ToolError = _calculator_module.ToolError
calculate = _calculator_module.calculate


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
