"""
Declarative tool registry: one governed spec per tool (name, category, description,
input contract, timeout, whether it reaches an external service, and the label the UI
shows). The executor, planner, approval gate and evaluation all read from here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Category = Literal["document", "video", "web", "finance", "calculator"]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    category: Category
    description: str
    input_description: str
    external: bool
    timeout_s: float
    label: str


TOOL_SPECS: dict[str, ToolSpec] = {
    spec.name: spec
    for spec in [
        ToolSpec("document_search", "document", "Retrieve passages from the user's uploaded PDFs.",
                 "natural-language query", False, 30.0, "Document retrieval"),
        ToolSpec("video_search", "video",
                 "Temporal retrieval over this conversation's video/audio transcripts (timestamps kept).",
                 "natural-language query (+ optional media_id from context)", False, 30.0, "Meeting evidence"),
        ToolSpec("media_insights", "video",
                 "Stored summary, key decisions, action items and open questions of this conversation's media.",
                 "comma-separated sections", False, 10.0, "Meeting insights"),
        ToolSpec("web_search", "web", "DuckDuckGo web search with source URLs.",
                 "search query (≤300 chars)", True, 20.0, "Web research"),
        ToolSpec("news_search", "web", "DuckDuckGo news search with publisher and date.",
                 "search query (≤300 chars)", True, 20.0, "News"),
        ToolSpec("stock_quote", "finance", "Latest price and daily change for a ticker.",
                 "ticker symbol, e.g. AAPL", True, 15.0, "Market data"),
        ToolSpec("price_history", "finance",
                 "Daily closes plus return, volatility, drawdown and moving averages.",
                 "ticker symbol + period (5d,1mo,3mo,6mo,1y)", True, 15.0, "Price history"),
        ToolSpec("calculator", "calculator", "Exact arithmetic via a safe AST evaluator.",
                 "arithmetic expression", False, 2.0, "Calculation"),
    ]
}


def tool_category(name: str) -> str | None:
    spec = TOOL_SPECS.get(name)
    return spec.category if spec else None
