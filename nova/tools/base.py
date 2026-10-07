"""
Tool governance primitives.

Every Nova tool returns a normalised `ToolResult`:

    {"tool": "stock_quote", "status": "success" | "error", "data": {...},
     "source": "...", "timestamp": "...", "latency_ms": 12.3, "error": None}

`run_tool` adds timing, logging and conversion of unexpected exceptions into
error results, so one failing tool never crashes the agent.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Literal

import requests
from pydantic import BaseModel, Field

from nova.config import get_settings
from nova.observability import log_event, metrics


class ToolError(Exception):
    """Expected, user-presentable tool failure (bad input, provider unavailable...)."""

    def __init__(self, user_message: str, detail: str | None = None):
        super().__init__(user_message)
        self.user_message = user_message
        self.detail = detail or user_message


class ToolResult(BaseModel):
    tool: str
    status: Literal["success", "error"]
    data: dict[str, Any] = Field(default_factory=dict)
    source: str | None = None
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    latency_ms: float = 0.0
    error: str | None = None
    input: dict[str, Any] = Field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "success"


def run_tool(
    name: str,
    fn: Callable[..., tuple[dict[str, Any], str | None]],
    **kwargs: Any,
) -> ToolResult:
    """Execute `fn(**kwargs) -> (data, source)` and normalise the outcome."""
    start = time.perf_counter()
    try:
        data, source = fn(**kwargs)
        result = ToolResult(tool=name, status="success", data=data, source=source, input=kwargs)
    except ToolError as exc:
        result = ToolResult(tool=name, status="error", error=exc.user_message, input=kwargs)
        log_event("tool_error", level=30, tool=name, detail=exc.detail)
    except Exception as exc:  # noqa: BLE001 - convert any failure into a structured error
        result = ToolResult(
            tool=name, status="error", error=f"{name} is temporarily unavailable.", input=kwargs
        )
        log_event("tool_exception", level=40, tool=name, error=repr(exc), exc_info=True)
    result.latency_ms = round((time.perf_counter() - start) * 1000, 1)
    metrics.incr(f"tool.{name}.{result.status}")
    metrics.observe(f"tool.{name}", result.latency_ms)
    log_event("tool_completed", tool=name, status=result.status, latency_ms=result.latency_ms)
    return result


_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def http_get_json(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
    session: requests.Session | None = None,
) -> Any:
    """GET with timeout and bounded exponential backoff on network errors / 429 / 5xx."""
    settings = get_settings()
    timeout = timeout or settings.http_timeout_s
    retries = settings.http_max_retries if max_retries is None else max_retries
    getter = session.get if session else requests.get
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = getter(url, params=params, headers=headers, timeout=timeout)
            if resp.status_code in _RETRYABLE_STATUS and attempt < retries:
                raise requests.HTTPError(f"retryable status {resp.status_code}", response=resp)
            resp.raise_for_status()
            return resp.json()
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            last_exc = exc
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status is not None and status not in _RETRYABLE_STATUS:
                break
            if attempt < retries:
                time.sleep(min(8.0, (2**attempt) * 0.5) + random.uniform(0, 0.25))
        except ValueError as exc:  # invalid JSON
            last_exc = exc
            break
    raise ToolError("The data provider is temporarily unavailable.", detail=repr(last_exc))
