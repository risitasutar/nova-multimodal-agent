"""Evaluation workers must print non-ASCII model text even when the console code page is not UTF-8."""

import subprocess
import sys
from pathlib import Path

from evaluation import worker_env

ROOT = Path(__file__).resolve().parents[2]
SNIPPET = "from evaluation import utf8_stdout; utf8_stdout(); print('\u20b9 10 crore \u2013 \u2713')"


def _run(env):
    return subprocess.run([sys.executable, "-c", SNIPPET], capture_output=True, cwd=ROOT, env=env, timeout=60)


def test_worker_stdout_is_utf8_even_with_cp1252_console():
    proc = _run({**worker_env(), "PYTHONIOENCODING": "cp1252", "PYTHONUTF8": "0"})
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    assert proc.stdout.decode("utf-8").strip() == "\u20b9 10 crore \u2013 \u2713"


def test_worker_env_forces_utf8():
    env = worker_env()
    assert env["PYTHONIOENCODING"] == "utf-8" and env["PYTHONUTF8"] == "1"
