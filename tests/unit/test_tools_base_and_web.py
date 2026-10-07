import pytest

from nova.config import get_settings
from nova.tools import web
from nova.tools.base import ToolError, ToolResult, run_tool
from nova.tools.registry import TOOL_SPECS


def test_run_tool_normalises_success_and_errors():
    ok = run_tool("demo", lambda x: ({"x": x}, "src"), x=1)
    assert isinstance(ok, ToolResult) and ok.ok and ok.data == {"x": 1} and ok.source == "src"
    assert ok.latency_ms >= 0 and ok.error is None

    def expected_failure(x):
        raise ToolError("Nice message", detail="internal detail")

    err = run_tool("demo", expected_failure, x=1)
    assert err.status == "error" and err.error == "Nice message"

    def crash(x):
        raise RuntimeError("boom with secret internals")

    crashed = run_tool("demo", crash, x=1)
    assert crashed.status == "error" and "boom" not in crashed.error  # internals never surface


def test_every_tool_is_governed():
    for name, spec in TOOL_SPECS.items():
        assert spec.name == name and spec.description and spec.input_description
        assert spec.timeout_s > 0 and spec.label
        assert spec.category in {"document", "video", "web", "finance", "calculator"}


def test_web_search_normalises_results(monkeypatch):
    raw = [
        {"title": "A", "href": "https://www.example.com/a", "body": "x" * 1000},
        {"title": "bad", "href": "javascript:alert(1)", "body": "nope"},
    ]
    monkeypatch.setattr(web, "_ddgs_call", lambda kind, q, n: raw)
    r = web.web_search("query")
    assert r.ok and len(r.data["results"]) == 1
    item = r.data["results"][0]
    assert item["domain"] == "example.com" and len(item["snippet"]) == web.MAX_SNIPPET_CHARS


def test_web_search_no_results_and_disabled(monkeypatch):
    monkeypatch.setattr(web, "_ddgs_call", lambda kind, q, n: [])
    assert web.web_search("q").status == "error"
    monkeypatch.setenv("NOVA_SEARCH_ENABLED", "false")
    get_settings.cache_clear()
    disabled = web.news_search("q")
    assert disabled.status == "error" and "disabled" in disabled.error


def test_ddgs_failure_degrades_gracefully(monkeypatch):
    class Broken:
        def __init__(self, *a, **k):
            pass

        def text(self, *a, **k):
            raise RuntimeError("ratelimit")

    import ddgs

    monkeypatch.setattr(ddgs, "DDGS", Broken)
    monkeypatch.setattr(web.time, "sleep", lambda s: None)
    r = web.web_search("anything")
    assert r.status == "error" and "temporarily unavailable" in r.error


@pytest.mark.parametrize("q", ["", "   "])
def test_empty_query_rejected(q):
    assert web.web_search(q).status == "error"
