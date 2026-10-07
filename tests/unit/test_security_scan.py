"""Repository hygiene: no hard-coded secrets or machine-specific paths in tracked source."""

import hashlib
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCANNED = [p for ext in ("*.py", "*.toml", "*.yml", "*.yaml", "*.bat", "*.md", "*.txt", "*.json", "Dockerfile",
                         ".env.example")
           for p in ROOT.rglob(ext)
           if not any(part in {".venv", "data", "logs", ".git", "__pycache__", ".mypy_cache", ".ruff_cache"}
                      for part in p.parts)]

PATTERNS = {
    "OpenAI-style key": re.compile(r"\bsk-[A-Za-z0-9]{20,}"),
    "LangSmith key": re.compile(r"\blsv2_[A-Za-z0-9_]{20,}"),
    "GitHub token": re.compile(r"\bghp_[A-Za-z0-9]{30,}"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY"),
    "macOS user path": re.compile(r"/Users/[a-z]+/Desktop"),
    "Windows user path": re.compile(r"[A-Za-z]:\\\\?Users\\\\?[A-Za-z]+\\\\?Desktop", re.I),
}
ALLOWED = {"test_security_scan.py"}

# The Alpha Vantage key leaked in the upstream tutorial is detected by hash, so the literal
# never has to appear in this repository (not even in this test).
_LEAKED_KEY_SHA256 = "1e4e6769358e9e3bfbd480c45dc8711ead55ba02c6a37e6774b964c7955eb1b4"
_KEY_SHAPED = re.compile(r"\b[A-Z0-9]{16}\b")


@pytest.mark.parametrize("name,pattern", PATTERNS.items())
def test_no_secrets_or_personal_paths(name, pattern):
    offenders = [
        str(p.relative_to(ROOT))
        for p in SCANNED
        if p.name not in ALLOWED and pattern.search(p.read_text(encoding="utf-8", errors="ignore"))
    ]
    assert not offenders, f"{name} found in: {offenders}"


def test_leaked_upstream_key_absent():
    offenders = [
        str(p.relative_to(ROOT))
        for p in SCANNED
        if any(hashlib.sha256(tok.encode()).hexdigest() == _LEAKED_KEY_SHA256
               for tok in _KEY_SHAPED.findall(p.read_text(encoding="utf-8", errors="ignore")))
    ]
    assert not offenders, f"leaked upstream Alpha Vantage key found in: {offenders}"


def test_env_file_is_gitignored():
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in ignore
