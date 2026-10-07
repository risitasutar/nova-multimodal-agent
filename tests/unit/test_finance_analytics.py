import math

import pytest

from nova.tools.finance.analytics import (
    annualized_volatility_pct,
    max_drawdown_pct,
    moving_average,
    period_return_pct,
    summarize,
)


def test_period_return():
    assert period_return_pct([100, 110]) == pytest.approx(10.0)
    assert period_return_pct([100]) is None


def test_max_drawdown_finds_peak_to_trough():
    # peak 120 → trough 90 = -25%
    assert max_drawdown_pct([100, 120, 110, 90, 130]) == pytest.approx(-25.0)
    assert max_drawdown_pct([1, 2, 3]) == 0.0


def test_volatility_matches_definition():
    closes = [100, 101, 99, 102, 100]
    logs = [math.log(b / a) for a, b in zip(closes, closes[1:])]
    mean = sum(logs) / len(logs)
    sd = math.sqrt(sum((x - mean) ** 2 for x in logs) / (len(logs) - 1))
    assert annualized_volatility_pct(closes) == pytest.approx(sd * math.sqrt(252) * 100)
    assert annualized_volatility_pct([100, 101]) is None


def test_moving_average_requires_enough_points():
    assert moving_average([1, 2, 3, 4], 2) == 3.5
    assert moving_average([1, 2], 5) is None


def test_summarize_is_complete_and_rounded():
    closes = [100.0, 102.0, 101.0, 105.0]
    dates = ["d1", "d2", "d3", "d4"]
    s = summarize(closes, dates)
    assert s["period_return_pct"] == 5.0
    assert s["best_day"]["date"] == "d4" and s["worst_day"]["date"] == "d3"
    assert s["trading_days"] == 4 and s["sma_20"] is None
