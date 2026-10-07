"""
Web and news search via DuckDuckGo (`ddgs`). No API key required.

Results keep their URLs so answers can cite them. Snippets are untrusted third-party
text: they are passed to the model as data, never as instructions.
"""

from __future__ import annotations

import time
from typing import Any
from urllib.parse import urlparse

from nova.config import get_settings
from nova.tools.base import ToolError, ToolResult, run_tool

MAX_SNIPPET_CHARS = 350


def _domain(url: str) -> str:
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def _ddgs_call(kind: str, query: str, max_results: int) -> list[dict[str, Any]]:
    from ddgs import DDGS  # imported lazily: optional at import time, heavy

    settings = get_settings()
    last: Exception | None = None
    for attempt in range(settings.http_max_retries + 1):
        try:
            client = DDGS(timeout=int(settings.http_timeout_s))
            if kind == "news":
                return list(client.news(query, region=settings.search_region, max_results=max_results))
            return list(client.text(query, region=settings.search_region, max_results=max_results))
        except Exception as exc:  # noqa: BLE001 - ddgs raises several exception types
            last = exc
            if "no results" in str(exc).lower():
                return []
            if attempt < settings.http_max_retries:
                time.sleep(0.5 * 2**attempt)
    raise ToolError("Web search is temporarily unavailable.", detail=repr(last))


def _normalise(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for item in raw:
        url = item.get("href") or item.get("url") or ""
        if not url.startswith(("http://", "https://")):
            continue
        results.append(
            {
                "title": (item.get("title") or _domain(url))[:200],
                "url": url,
                "domain": _domain(url),
                "snippet": (item.get("body") or "")[:MAX_SNIPPET_CHARS],
                "date": item.get("date"),
                "publisher": item.get("source"),
            }
        )
    return results


def _search(query: str, kind: str) -> tuple[dict[str, Any], str | None]:
    settings = get_settings()
    if not settings.search_enabled:
        raise ToolError("Web search is disabled in this deployment.")
    query = query.strip()[:300]
    if not query:
        raise ToolError("Empty search query.")
    results = _normalise(_ddgs_call(kind, query, settings.search_max_results))
    if not results:
        raise ToolError(f"No web results found for '{query}'.")
    return {"query": query, "results": results}, "duckduckgo"


def web_search(query: str) -> ToolResult:
    return run_tool("web_search", lambda query: _search(query, "text"), query=query)


def news_search(query: str) -> ToolResult:
    return run_tool("news_search", lambda query: _search(query, "news"), query=query)
