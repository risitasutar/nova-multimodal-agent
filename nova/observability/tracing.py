"""
Optional LangSmith tracing.

Tracing is enabled only when LANGCHAIN_TRACING_V2=true *and* LANGCHAIN_API_KEY is set.
Otherwise it is explicitly disabled so a stray env var can never make requests fail or
leak data to an external service.
"""

from __future__ import annotations

import os

from nova.config import Settings
from nova.observability.logging import log_event


def configure_tracing(settings: Settings) -> bool:
    key = settings.langchain_api_key.get_secret_value() if settings.langchain_api_key else ""
    enabled = bool(settings.langchain_tracing_v2 and key)
    if enabled:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = key
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        os.environ.pop("LANGSMITH_TRACING", None)
    log_event("tracing_configured", langsmith_enabled=enabled, project=settings.langchain_project)
    return enabled
