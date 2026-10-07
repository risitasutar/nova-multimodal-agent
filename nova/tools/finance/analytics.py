"""
Deterministic price analytics. Pure functions over a list of closing prices — no LLM,
so every number Nova reports about a price series is reproducible and testable.
"""

from __future__ import annotations

import math
import statistics

TRADING_DAYS = 252


def simple_returns(closes: list[float]) -> list[float]:
    return [(b / a) - 1 for a, b in zip(closes, closes[1:]) if a]


def period_return_pct(closes: list[float]) -> float | None:
    if len(closes) < 2 or not closes[0]:
        return None
    return (closes[-1] / closes[0] - 1) * 100


def annualized_volatility_pct(closes: list[float]) -> float | None:
    """Std-dev of daily log returns × √252."""
    logs = [math.log(b / a) for a, b in zip(closes, closes[1:]) if a > 0 and b > 0]
    if len(logs) < 2:
        return None
    return statistics.stdev(logs) * math.sqrt(TRADING_DAYS) * 100


def moving_average(closes: list[float], window: int) -> float | None:
    if window <= 0 or len(closes) < window:
        return None
    return sum(closes[-window:]) / window


def max_drawdown_pct(closes: list[float]) -> float | None:
    """Largest peak-to-trough decline, as a negative percentage."""
    if len(closes) < 2:
        return None
    peak, worst = closes[0], 0.0
    for price in closes:
        peak = max(peak, price)
        if peak:
            worst = min(worst, price / peak - 1)
    return worst * 100


def summarize(closes: list[float], dates: list[str]) -> dict:
    """Risk/return summary of a closing-price series."""
    rets = simple_returns(closes)

    def r(x: float | None, nd: int = 2) -> float | None:
        return None if x is None else round(x, nd)

    best = max(range(len(rets)), key=lambda i: rets[i]) if rets else None
    worst = min(range(len(rets)), key=lambda i: rets[i]) if rets else None
    return {
        "start_date": dates[0] if dates else None,
        "end_date": dates[-1] if dates else None,
        "start_price": r(closes[0]) if closes else None,
        "end_price": r(closes[-1]) if closes else None,
        "period_high": r(max(closes)) if closes else None,
        "period_low": r(min(closes)) if closes else None,
        "period_return_pct": r(period_return_pct(closes)),
        "annualized_volatility_pct": r(annualized_volatility_pct(closes)),
        "max_drawdown_pct": r(max_drawdown_pct(closes)),
        "sma_20": r(moving_average(closes, 20)),
        "sma_50": r(moving_average(closes, 50)),
        "best_day": {"date": dates[best + 1], "return_pct": r(rets[best] * 100)} if best is not None else None,
        "worst_day": {"date": dates[worst + 1], "return_pct": r(rets[worst] * 100)} if worst is not None else None,
        "trading_days": len(closes),
    }
