"""Safe arithmetic: an AST whitelist evaluator. `eval` is never used."""

from __future__ import annotations

import ast
import math
import operator
import re
from collections.abc import Callable
from typing import Any

from nova.tools.base import ToolError, ToolResult, run_tool

_BIN_OPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type, Callable[[Any], Any]] = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS: dict[str, Callable[..., float]] = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "log": math.log,
    "exp": math.exp,
    "min": min,
    "max": max,
}
MAX_EXPRESSION_CHARS = 300
MAX_EXPONENT = 1000
MAX_MAGNITUDE = 1e100


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ValueError("exponent too large")
        value = _BIN_OPS[type(node.op)](left, right)
    elif isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        value = _UNARY_OPS[type(node.op)](_eval(node.operand))
    elif (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCS
        and not node.keywords
        and len(node.args) <= 4
    ):
        value = _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    else:
        raise ValueError("only numbers, + - * / // % **, parentheses and sqrt/abs/round/log/exp/min/max are allowed")
    if isinstance(value, complex) or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        raise ValueError("result is not a finite real number")
    if abs(value) > MAX_MAGNITUDE:
        raise ValueError("result too large")
    return value


def normalize_expression(expression: str) -> str:
    expr = expression.strip()
    expr = expr.replace("^", "**").replace("×", "*").replace("÷", "/").replace("−", "-")
    expr = re.sub(r"(?<=\d),(?=\d{3}\b)", "", expr)  # 1,234 -> 1234
    expr = re.sub(r"(\d+(?:\.\d+)?)\s*%(?!\s*[\d(])", r"(\1/100)", expr)  # 5% -> (5/100); 17 % 5 stays modulo
    return expr.rstrip("=?. ")


def evaluate(expression: str) -> float | int:
    """Evaluate an arithmetic expression; raises ToolError on invalid input."""
    if not expression or len(expression) > MAX_EXPRESSION_CHARS:
        raise ToolError("Expression is empty or too long.")
    expr = normalize_expression(expression)
    try:
        result = _eval(ast.parse(expr, mode="eval"))
    except ZeroDivisionError as exc:
        raise ToolError("Division by zero is not allowed.") from exc
    except (SyntaxError, ValueError, TypeError, OverflowError) as exc:
        raise ToolError(f"Could not evaluate '{expression}': {exc}") from exc
    if isinstance(result, float) and result.is_integer() and abs(result) < 1e15:
        return int(result)
    return result


def calculate(expression: str, label: str | None = None) -> ToolResult:
    def _run(expression: str, label: str | None) -> tuple[dict[str, Any], str | None]:
        value = evaluate(expression)
        rounded = round(value, 6) if isinstance(value, float) else value
        return {"expression": expression, "result": rounded, "label": label}, "calculator"

    return run_tool("calculator", _run, expression=expression, label=label)


_PURE_EXPR = re.compile(r"^[\d\s\.\,\+\-\*/\^\(\)%×÷−]+$")
_LEADING_WORDS = re.compile(
    r"^(please\s+)?(what\s+is|what's|calculate|compute|evaluate|solve|how\s+much\s+is)\s+", re.I
)


def extract_pure_expression(text: str) -> str | None:
    """Return the expression if `text` is just arithmetic (e.g. 'What is 9283 * 47?')."""
    candidate = _LEADING_WORDS.sub("", text.strip()).rstrip("?.! =")
    if not candidate or not _PURE_EXPR.match(candidate):
        return None
    if not re.search(r"\d\s*[\+\-\*/\^×÷%]\s*\(?\s*\d", candidate):
        return None
    try:
        evaluate(candidate)
    except ToolError:
        return None
    return candidate
