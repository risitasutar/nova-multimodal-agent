import pytest

from nova.tools.base import ToolError
from nova.tools.calculator import calculate, evaluate, extract_pure_expression


@pytest.mark.parametrize(
    "expr,expected",
    [
        ("9283 * 47", 436301),
        ("(2 + 3) ** 2", 25),
        ("7289 * 347 / 17", pytest.approx(148781.3529, rel=1e-6)),
        ("2^10", 1024),
        ("1,250.5 + 1,000", 2250.5),
        ("10000 * (1 + 7%) ** 10", pytest.approx(19671.51, abs=0.01)),
        ("-5 + 3", -2),
        ("sqrt(16) + abs(-2)", 6),
        ("17 // 5 + 17 % 5", 5),
        ("6 × 7 ÷ 2", 21),
    ],
)
def test_evaluates_arithmetic(expr, expected):
    assert evaluate(expr) == expected


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('echo pwned')",
        "open('secrets.txt')",
        "(1).__class__",
        "[1, 2, 3]",
        "x + 1",
        "2 ** 100000",
        "9 ** 9 ** 9",
        "lambda: 1",
        "",
        "1" * 400,
    ],
)
def test_rejects_unsafe_or_invalid_expressions(expr):
    with pytest.raises(ToolError):
        evaluate(expr)


def test_division_by_zero_is_a_tool_error():
    with pytest.raises(ToolError, match="Division by zero"):
        evaluate("1 / 0")


def test_calculate_returns_normalised_result():
    r = calculate("2 + 2", label="sum")
    assert r.ok and r.tool == "calculator"
    assert r.data == {"expression": "2 + 2", "result": 4, "label": "sum"}
    bad = calculate("1/0")
    assert bad.status == "error" and "Division by zero" in bad.error and bad.data == {}


@pytest.mark.parametrize(
    "text,expected",
    [
        ("What is 9283 * 47?", "9283 * 47"),
        ("calculate (1250.75 - 318.2) / 4.5", "(1250.75 - 318.2) / 4.5"),
        ("7289*347/17", "7289*347/17"),
        ("What is Apple's revenue?", None),
        ("2025", None),
        ("What was revenue in 2024 vs 2025?", None),
    ],
)
def test_extract_pure_expression(text, expected):
    assert extract_pure_expression(text) == expected
