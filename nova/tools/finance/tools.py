"""Finance tools exposed to the agent. Read-only analytics — no trading, no advice."""

from __future__ import annotations

from typing import Any

from nova.tools.base import ToolError, ToolResult, run_tool
from nova.tools.finance.analytics import summarize
from nova.tools.finance.providers import PERIODS, get_provider, normalize_symbol

MAX_SERIES_POINTS = 260


def stock_quote(symbol: str) -> ToolResult:
    def _run(symbol: str) -> tuple[dict[str, Any], str | None]:
        provider = get_provider()
        quote = provider.quote(normalize_symbol(symbol))
        return quote.to_dict(), f"{quote.provider} ({quote.symbol} quote, {quote.as_of})"

    return run_tool("stock_quote", _run, symbol=symbol)


def price_history(symbol: str, period: str = "1mo") -> ToolResult:
    def _run(symbol: str, period: str) -> tuple[dict[str, Any], str | None]:
        if period not in PERIODS:
            raise ToolError(f"Unsupported period '{period}'. Use one of: {', '.join(PERIODS)}.")
        hist = get_provider().history(normalize_symbol(symbol), period)
        if len(hist.closes) < 2:
            raise ToolError(f"Not enough price history for {hist.symbol}.")
        data = {
            "symbol": hist.symbol,
            "name": hist.name,
            "period": period,
            "currency": hist.currency,
            "provider": hist.provider,
            "analytics": summarize(hist.closes, hist.dates),
            "series": {
                "dates": hist.dates[-MAX_SERIES_POINTS:],
                "closes": [round(c, 2) for c in hist.closes[-MAX_SERIES_POINTS:]],
            },
        }
        source = f"{hist.provider} ({hist.symbol} daily closes, {hist.dates[0]} to {hist.dates[-1]})"
        return data, source

    return run_tool("price_history", _run, symbol=symbol, period=period)
