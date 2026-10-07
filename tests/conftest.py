"""
Shared fixtures. Every test runs with an isolated data directory and settings;
no test in the default suite touches Ollama, the network or the real database.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import pytest

from nova.config import get_settings


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("NOVA_ENV", "test")
    monkeypatch.setenv("NOVA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("NOVA_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("NOVA_HTTP_MAX_RETRIES", "1")
    for var in ("ALPHAVANTAGE_API_KEY", "NOVA_API_KEY", "LANGCHAIN_API_KEY", "NOVA_APPROVAL_REQUIRED"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def report_bytes() -> bytes:
    from evaluation.fixtures.documents import fixture_path

    return fixture_path("northwind_annual_report_2025.pdf").read_bytes()


@pytest.fixture
def injection_bytes() -> bytes:
    from evaluation.fixtures.documents import fixture_path

    return fixture_path("vendor_contract_injection.pdf").read_bytes()


def fake_tools(overrides: dict[str, Callable] | None = None) -> dict[str, Callable]:
    """Network-free tool implementations returning normalised ToolResults."""
    from nova.tools.base import ToolResult
    from nova.tools.calculator import calculate

    def stock_quote(symbol: str) -> ToolResult:
        return ToolResult(tool="stock_quote", status="success", source=f"TestFeed ({symbol} quote, 2026-10-02)",
                          input={"symbol": symbol},
                          data={"symbol": symbol, "name": f"{symbol} Inc.", "price": 333.69, "previous_close": 330.32,
                                "change": 3.37, "change_percent": 1.02, "currency": "USD", "as_of": "2026-10-02",
                                "day_high": 334.54, "day_low": 330.61, "fifty_two_week_high": 345.34,
                                "fifty_two_week_low": 243.42, "provider": "TestFeed"})

    def price_history(symbol: str, period: str = "1mo") -> ToolResult:
        from nova.tools.finance.analytics import summarize

        closes = [100.0, 102.0, 101.0, 105.0, 103.0, 108.0]
        dates = [f"2026-09-{d:02d}" for d in range(1, 7)]
        return ToolResult(tool="price_history", status="success", source=f"TestFeed ({symbol} daily closes)",
                          input={"symbol": symbol, "period": period},
                          data={"symbol": symbol, "name": None, "period": period, "currency": "USD",
                                "provider": "TestFeed", "analytics": summarize(closes, dates),
                                "series": {"dates": dates, "closes": closes}})

    def web_search(query: str) -> ToolResult:
        return ToolResult(tool="web_search", status="success", source="duckduckgo", input={"query": query},
                          data={"query": query, "results": [
                              {"title": "Eiffel Tower - Wikipedia", "url": "https://en.wikipedia.org/wiki/Eiffel_Tower",
                               "domain": "en.wikipedia.org", "snippet": "The tower was completed in 1889.",
                               "date": None, "publisher": None}]})

    def news_search(query: str) -> ToolResult:
        return ToolResult(tool="news_search", status="success", source="duckduckgo", input={"query": query},
                          data={"query": query, "results": [
                              {"title": "Chipmaker shares rise", "url": "https://example.com/news/1",
                               "domain": "example.com", "snippet": "Shares rose 3% after earnings.",
                               "date": "2026-10-01", "publisher": "Example News"}]})

    tools: dict[str, Callable] = {
        "stock_quote": stock_quote, "price_history": price_history, "web_search": web_search,
        "news_search": news_search, "calculator": calculate,
    }
    tools.update(overrides or {})
    return tools


def fake_transcriber(audio, language, settings):
    """Deterministic stand-in for Whisper: a transcript fixture, regardless of the audio."""
    from evaluation.fixtures.media import load_transcript

    return load_transcript("quarterly_review_meeting")


@pytest.fixture
def make_service(tmp_path):
    """Factory: a NovaService wired to scripted LLM, hashing embeddings and fake tools."""
    from nova.agent.graph import AgentDeps
    from nova.memory import ThreadRegistry, connect, make_checkpointer
    from nova.rag.retrieve import Retriever
    from nova.rag.store import DocumentStore
    from nova.service import NovaService
    from tests.fakes import HashEmbeddings, ScriptedChatModel

    created: list[Any] = []

    def _make(answer_fn=None, structured_fn=None, tools=None, min_relevance: float = 0.25, llm=None,
              transcriber=None, video_min_relevance: float = 0.2):
        s = get_settings()
        emb = HashEmbeddings()
        store = DocumentStore(s.vector_dir, emb, "hash-embeddings")
        retriever = Retriever(store, emb, top_k=s.retrieval_top_k, candidates=s.retrieval_candidates,
                              min_relevance=min_relevance)
        model = llm or ScriptedChatModel(answer_fn=answer_fn, structured_fn=structured_fn)
        from nova.video.service import MediaService
        from nova.video.storage import MediaRegistry

        media_store = DocumentStore(s.media_vector_dir, emb, "hash-embeddings")
        media = MediaService(s, MediaRegistry(connect(s.db_path)), media_store,
                             Retriever(media_store, emb, top_k=s.retrieval_top_k, candidates=s.retrieval_candidates,
                                       min_relevance=video_min_relevance),
                             model, transcriber=transcriber or fake_transcriber)
        svc = NovaService(AgentDeps(llm=model, store=store, retriever=retriever, settings=s,
                                    tools=fake_tools(tools), media=media),
                          make_checkpointer(connect(s.db_path)), ThreadRegistry(connect(s.db_path)))
        created.append(svc)
        return svc

    yield _make


def last_user_content(messages) -> str:
    return str(messages[-1].content)


os.environ.setdefault("PYTHONIOENCODING", "utf-8")
