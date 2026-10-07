"""
Curated demo flow. Queries only — every answer is produced live by the real agent,
tools and data (nothing is pre-recorded). The sample report is the fictional
evaluation fixture, so document answers can be checked against the PDF.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nova.config import PROJECT_ROOT

SAMPLE_REPORT = PROJECT_ROOT / "evaluation" / "fixtures" / "northwind_annual_report_2025.pdf"

DEMO_QUERIES: list[tuple[str, str, str]] = [
    ("💬", "General", "What does a company's gross margin tell an investor?"),
    ("📄", "Document", "What were Northwind's main risk factors in 2025?"),
    ("📈", "Live finance", "What is the current stock price of AAPL?"),
    ("🧭", "Multi-tool", "Analyze NVDA over the last month and summarize major news."),
    ("🧮", "Calculation", "What is 9283 * 47?"),
    ("🚫", "Unsupported", "What dividend per share did Northwind pay in 2025?"),
    ("🧠", "Memory", "Which of the figures you gave me so far was the largest?"),
    ("🎬", "Meeting", "What were the key decisions in the meeting?"),
    ("⏱", "Timestamp", "When did they discuss the hiring plan?"),
    ("🔀", "Cross-source", "Is the revenue target in the meeting consistent with the strategy document?"),
]


def sample_meeting_transcript() -> tuple[str, Any] | None:
    """Fictional Q3 review meeting (timestamped transcript fixture) for the demo."""
    try:
        from evaluation.fixtures.media import load_transcript, media_filename
    except ImportError:
        return None
    return media_filename("quarterly_review_meeting"), load_transcript("quarterly_review_meeting")


def sample_strategy() -> tuple[str, bytes] | None:
    """Fictional strategy PDF that deliberately conflicts with the sample meeting (cross-source demo)."""
    try:
        from evaluation.fixtures.media import strategy_pdf
    except ImportError:
        return None
    return "strategy_2026.pdf", strategy_pdf()


def sample_report() -> tuple[str, bytes] | None:
    path: Path = SAMPLE_REPORT
    if not path.exists():
        return None
    return path.name, path.read_bytes()
