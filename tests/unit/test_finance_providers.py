"""Provider parsing, fallback and failure handling with mocked HTTP (no network)."""

import pytest
import requests

from nova.config import get_settings
from nova.tools.base import ToolError
from nova.tools.finance import providers
from nova.tools.finance.providers import (
    AlphaVantageProvider,
    FallbackProvider,
    YahooChartProvider,
    get_provider,
    normalize_symbol,
)
from nova.tools.finance.tools import price_history, stock_quote

YAHOO = {
    "chart": {
        "result": [
            {
                "meta": {"symbol": "AAPL", "currency": "USD", "regularMarketPrice": 333.69,
                         "regularMarketTime": 1790971201, "longName": "Apple Inc.",
                         "fiftyTwoWeekHigh": 345.34, "fiftyTwoWeekLow": 243.42},
                "timestamp": [1790602200, 1790688600, 1790775000, 1790861400, 1790947800],
                "indicators": {"quote": [{"close": [338.4, 329.4, None, 330.32, 333.69]}]},
            }
        ],
        "error": None,
    }
}


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload, self.status_code = payload, status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}", response=self)


@pytest.fixture
def http(monkeypatch):
    calls = []
    queue: list = []

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append((url, params))
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr("nova.tools.base.time.sleep", lambda s: None)
    return queue, calls


def test_yahoo_quote_parses_and_skips_null_closes(http):
    queue, _ = http
    queue.append(FakeResponse(YAHOO))
    q = YahooChartProvider().quote("AAPL")
    assert q.price == 333.69 and q.previous_close == 330.32
    assert q.change == pytest.approx(3.37, abs=1e-6) and q.name == "Apple Inc."


def test_unknown_symbol_is_reported(http):
    queue, _ = http
    queue.append(FakeResponse({"chart": {"result": None, "error": {"code": "Not Found"}}}))
    with pytest.raises(ToolError, match="No market data"):
        YahooChartProvider().quote("ZZZZQQ")


def test_network_errors_retry_then_fail_gracefully(http):
    queue, calls = http
    queue.extend([requests.ConnectionError("down"), requests.ConnectionError("down")])
    result = stock_quote("AAPL")
    assert result.status == "error" and "temporarily unavailable" in result.error
    assert len(calls) == 2  # 1 try + NOVA_HTTP_MAX_RETRIES=1


def test_alpha_vantage_rate_limit_falls_back_to_yahoo(http):
    queue, calls = http
    queue.append(FakeResponse({"Information": "rate limit reached"}))
    queue.append(FakeResponse(YAHOO))
    q = FallbackProvider([AlphaVantageProvider("k"), YahooChartProvider()]).quote("AAPL")
    assert q.provider == "Yahoo Finance"
    assert "alphavantage" in calls[0][0] and "yahoo" in calls[1][0]


def test_unknown_symbol_does_not_fall_back(http):
    queue, calls = http
    queue.append(FakeResponse({"Error Message": "Invalid API call"}))
    with pytest.raises(ToolError, match="No market data"):
        FallbackProvider([AlphaVantageProvider("k"), YahooChartProvider()]).quote("NOPE")
    assert len(calls) == 1


def test_missing_key_message_when_alphavantage_forced(monkeypatch):
    monkeypatch.setenv("NOVA_FINANCE_PROVIDER", "alphavantage")
    get_settings.cache_clear()
    with pytest.raises(ToolError, match="ALPHAVANTAGE_API_KEY"):
        get_provider()


def test_provider_selection(monkeypatch):
    assert isinstance(get_provider(), YahooChartProvider)  # no key → keyless provider
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "test-key")
    get_settings.cache_clear()
    assert isinstance(get_provider(), FallbackProvider)


def test_key_is_sent_as_param_not_logged(http, caplog):
    queue, calls = http
    queue.append(FakeResponse({"Global Quote": {"01. symbol": "IBM", "05. price": "10.0", "07. latest trading day": "2026-10-02"}}))
    AlphaVantageProvider("super-secret-key").quote("IBM")
    assert calls[0][1]["apikey"] == "super-secret-key"
    assert "super-secret-key" not in caplog.text


@pytest.mark.parametrize("bad", ["", "drop table", "AAPL;rm", "$$$", "A" * 20])
def test_symbol_validation(bad):
    with pytest.raises(ToolError):
        normalize_symbol(bad)


def test_price_history_tool_output_shape(http):
    queue, _ = http
    queue.append(FakeResponse(YAHOO))
    r = price_history("aapl", "1mo")
    assert r.ok and r.data["symbol"] == "AAPL" and r.data["analytics"]["trading_days"] == 4
    assert set(r.data["series"]) == {"dates", "closes"}
    assert price_history("AAPL", "10y").status == "error"


def test_no_hardcoded_key_in_provider_module():
    """The upstream tutorial's leaked Alpha Vantage key is matched by SHA-256, never stored here."""
    import hashlib
    import inspect

    from tests.unit.test_security_scan import _KEY_SHAPED, _LEAKED_KEY_SHA256

    source = inspect.getsource(providers)
    assert not any(hashlib.sha256(tok.encode()).hexdigest() == _LEAKED_KEY_SHA256
                   for tok in _KEY_SHAPED.findall(source))
