"""
Market-data providers behind one interface.

* AlphaVantageProvider – official API, requires ALPHAVANTAGE_API_KEY (free tier: 25 req/day).
* YahooChartProvider   – Yahoo Finance public chart endpoint, no key. Unofficial: it can
                         change or rate-limit without notice, so it is a fallback.

`get_provider()` in "auto" mode prefers Alpha Vantage when a key is configured and
falls back to Yahoo when Alpha Vantage is rate-limited or unavailable.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Protocol

from nova.config import get_settings
from nova.tools.base import ToolError, http_get_json

TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-^=]{0,14}$")
PERIODS = {"5d": 5, "1mo": 22, "3mo": 66, "6mo": 130, "1y": 252, "2y": 504, "5y": 1260}
_YAHOO_HEADERS = {"User-Agent": "Mozilla/5.0 (Nova research assistant)"}


def normalize_symbol(symbol: str) -> str:
    sym = (symbol or "").strip().upper().lstrip("$")
    if not TICKER_RE.match(sym):
        raise ToolError(f"'{symbol}' is not a valid ticker symbol.")
    return sym


@dataclass
class Quote:
    symbol: str
    price: float
    previous_close: float | None
    change: float | None
    change_percent: float | None
    currency: str | None
    as_of: str
    name: str | None = None
    day_high: float | None = None
    day_low: float | None = None
    fifty_two_week_high: float | None = None
    fifty_two_week_low: float | None = None
    provider: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PriceHistory:
    symbol: str
    dates: list[str]
    closes: list[float]
    currency: str | None
    provider: str
    name: str | None = None
    meta: dict = field(default_factory=dict)


class MarketDataProvider(Protocol):
    name: str

    def quote(self, symbol: str) -> Quote: ...

    def history(self, symbol: str, period: str) -> PriceHistory: ...


class ProviderRateLimited(ToolError):
    pass


def _iso(ts: int | float) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).date().isoformat()


class YahooChartProvider:
    name = "Yahoo Finance"
    BASE = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

    def _chart(self, symbol: str, period: str) -> dict:
        data = http_get_json(
            self.BASE.format(symbol=symbol),
            params={"range": period, "interval": "1d"},
            headers=_YAHOO_HEADERS,
        )
        chart = (data or {}).get("chart", {})
        if chart.get("error") or not chart.get("result"):
            raise ToolError(f"No market data found for '{symbol}'.", detail=str(chart.get("error")))
        return chart["result"][0]

    def history(self, symbol: str, period: str) -> PriceHistory:
        result = self._chart(symbol, period)
        meta = result.get("meta", {})
        closes_raw = (result.get("indicators", {}).get("quote") or [{}])[0].get("close") or []
        pairs = [
            (_iso(ts), float(c))
            for ts, c in zip(result.get("timestamp") or [], closes_raw)
            if c is not None
        ]
        if not pairs:
            raise ToolError(f"No price history available for '{symbol}'.")
        return PriceHistory(
            symbol=meta.get("symbol", symbol),
            dates=[d for d, _ in pairs],
            closes=[round(c, 4) for _, c in pairs],
            currency=meta.get("currency"),
            provider=self.name,
            name=meta.get("longName") or meta.get("shortName"),
            meta=meta,
        )

    def quote(self, symbol: str) -> Quote:
        hist = self.history(symbol, "5d")
        meta = hist.meta
        price = float(meta.get("regularMarketPrice") or hist.closes[-1])
        prev = hist.closes[-2] if len(hist.closes) >= 2 else meta.get("chartPreviousClose")
        change = round(price - prev, 4) if prev else None
        pct = round(change / prev * 100, 4) if prev and change is not None else None
        as_of = _iso(meta["regularMarketTime"]) if meta.get("regularMarketTime") else hist.dates[-1]
        return Quote(
            symbol=hist.symbol,
            price=round(price, 4),
            previous_close=round(prev, 4) if prev else None,
            change=change,
            change_percent=pct,
            currency=hist.currency,
            as_of=as_of,
            name=hist.name,
            day_high=meta.get("regularMarketDayHigh"),
            day_low=meta.get("regularMarketDayLow"),
            fifty_two_week_high=meta.get("fiftyTwoWeekHigh"),
            fifty_two_week_low=meta.get("fiftyTwoWeekLow"),
            provider=self.name,
        )


class AlphaVantageProvider:
    name = "Alpha Vantage"
    BASE = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ToolError("Market data is not configured (missing ALPHAVANTAGE_API_KEY).")
        self._key = api_key

    def _get(self, params: dict) -> dict:
        data = http_get_json(self.BASE, params={**params, "apikey": self._key})
        if not isinstance(data, dict):
            raise ToolError("Market data is temporarily unavailable.")
        note = data.get("Note") or data.get("Information")
        if note:
            raise ProviderRateLimited("Market data is temporarily unavailable (provider rate limit).", detail=note)
        if "Error Message" in data:
            raise ToolError("No market data found for that symbol.", detail=data["Error Message"])
        return data

    def quote(self, symbol: str) -> Quote:
        q = self._get({"function": "GLOBAL_QUOTE", "symbol": symbol}).get("Global Quote") or {}
        if not q.get("05. price"):
            raise ToolError(f"No market data found for '{symbol}'.")
        pct = (q.get("10. change percent") or "").rstrip("%")
        return Quote(
            symbol=q.get("01. symbol", symbol),
            price=float(q["05. price"]),
            previous_close=float(q["08. previous close"]) if q.get("08. previous close") else None,
            change=float(q["09. change"]) if q.get("09. change") else None,
            change_percent=float(pct) if pct else None,
            currency="USD",
            as_of=q.get("07. latest trading day", ""),
            day_high=float(q["03. high"]) if q.get("03. high") else None,
            day_low=float(q["04. low"]) if q.get("04. low") else None,
            provider=self.name,
        )

    def history(self, symbol: str, period: str) -> PriceHistory:
        outputsize = "compact" if PERIODS.get(period, 22) <= 100 else "full"
        series = self._get(
            {"function": "TIME_SERIES_DAILY", "symbol": symbol, "outputsize": outputsize}
        ).get("Time Series (Daily)")
        if not series:
            raise ToolError(f"No price history available for '{symbol}'.")
        days = sorted(series.items())[-(PERIODS.get(period, 22) + 1):]
        return PriceHistory(
            symbol=symbol,
            dates=[d for d, _ in days],
            closes=[float(v["4. close"]) for _, v in days],
            currency="USD",
            provider=self.name,
        )


class FallbackProvider:
    """Try providers in order; move on when one is rate-limited or unavailable."""

    def __init__(self, providers: list[MarketDataProvider]) -> None:
        self.providers = providers
        self.name = " → ".join(p.name for p in providers)

    def _try(self, method: str, *args: str):
        last: ToolError | None = None
        for provider in self.providers:
            try:
                return getattr(provider, method)(*args)
            except ToolError as exc:
                last = exc
                if not isinstance(exc, ProviderRateLimited) and "unavailable" not in exc.user_message:
                    raise  # e.g. unknown symbol: another provider will not help
        assert last is not None
        raise last

    def quote(self, symbol: str) -> Quote:
        return self._try("quote", symbol)

    def history(self, symbol: str, period: str) -> PriceHistory:
        return self._try("history", symbol, period)


def get_provider() -> MarketDataProvider:
    settings = get_settings()
    key = settings.alphavantage_api_key.get_secret_value() if settings.alphavantage_api_key else ""
    if settings.finance_provider == "alphavantage":
        return AlphaVantageProvider(key)
    if settings.finance_provider == "yahoo" or not key:
        return YahooChartProvider()
    return FallbackProvider([AlphaVantageProvider(key), YahooChartProvider()])
