"""Nova evaluation harnesses."""

from __future__ import annotations

import os
import sys


def worker_env() -> dict[str, str]:
    """Environment for case workers: force UTF-8 stdio (Windows pipes default to the ANSI code page)."""
    return {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}


def utf8_stdout() -> None:
    """Make this process's stdout/stderr UTF-8 so records with non-ASCII model text can be printed."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
