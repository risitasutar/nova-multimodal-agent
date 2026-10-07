"""
Structured JSON logging with request context and secret redaction.

`log_event("name", **fields)` writes one JSON line to stderr and `logs/nova.log`.
`request_context(request_id=..., thread_id=...)` binds IDs to every log line in scope.
"""

from __future__ import annotations

import contextvars
import json
import logging
import re
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from typing import Any

_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)
_thread_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("thread_id", default=None)

_SENSITIVE_KEY = re.compile(r"(api[_-]?key|token|secret|password|authorization|apikey)", re.I)
_SENSITIVE_VALUE = [
    re.compile(r"(apikey=)[^&\s\"']+", re.I),
    re.compile(r"\b(sk-|lsv2_|ls__)[A-Za-z0-9_\-]{8,}"),
    re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]+", re.I),
]
REDACTED = "***"

logger = logging.getLogger("nova")


def redact(value: Any) -> Any:
    """Recursively mask secrets in keys and string values."""
    if isinstance(value, dict):
        return {
            k: (REDACTED if isinstance(k, str) and _SENSITIVE_KEY.search(k) else redact(v))
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    if isinstance(value, str):
        for pattern in _SENSITIVE_VALUE:
            value = pattern.sub(lambda m: (m.group(1) if m.groups() else "") + REDACTED, value)
        return value
    return value


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        if rid := _request_id.get():
            payload["request_id"] = rid
        if tid := _thread_id.get():
            payload["thread_id"] = tid
        fields = getattr(record, "fields", None)
        if fields:
            payload.update(fields)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(redact(payload), default=str, ensure_ascii=False)


class _PlainFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = f"{record.levelname:<7} {record.name}: {record.getMessage()}"
        fields = getattr(record, "fields", None)
        if fields:
            base += " " + json.dumps(redact(fields), default=str)
        return base


_configured = False


def configure_logging(level: str = "INFO", json_logs: bool = True, log_dir: Any = None) -> None:
    """Idempotent logging setup (console + rotating file)."""
    global _configured
    if _configured:
        return
    logger.setLevel(level.upper())
    logger.propagate = False
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(JsonFormatter() if json_logs else _PlainFormatter())
    logger.addHandler(console)
    if log_dir is not None:
        file_handler = RotatingFileHandler(
            str(log_dir / "nova.log"), maxBytes=5_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(JsonFormatter())
        logger.addHandler(file_handler)
    _configured = True


def log_event(event: str, level: int = logging.INFO, exc_info: bool = False, **fields: Any) -> None:
    logger.log(level, event, extra={"fields": fields}, exc_info=exc_info)


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


def current_request_id() -> str | None:
    return _request_id.get()


@contextmanager
def request_context(request_id: str | None = None, thread_id: str | None = None) -> Iterator[str]:
    rid = request_id or _request_id.get() or new_request_id()
    t1 = _request_id.set(rid)
    t2 = _thread_id.set(thread_id) if thread_id else None
    try:
        yield rid
    finally:
        _request_id.reset(t1)
        if t2 is not None:
            _thread_id.reset(t2)
