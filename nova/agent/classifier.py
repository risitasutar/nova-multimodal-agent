"""
Query classification = deterministic fast paths + structured LLM output + policy guards.

1. Fast paths (no LLM call): pure arithmetic ("What is 9283 * 47?") and greetings.
2. Structured LLM classification (`QueryClassification` JSON schema).
3. Heuristic fallback if the LLM output is unusable.
4. Policy guards that override the model where a wrong route would cause hallucination:
   time-sensitive wording can never be routed to `general`; explicit references to the
   uploaded document go to RAG; finance without a ticker asks for clarification.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, field_validator

from nova.agent.prompts import CLASSIFIER_SYSTEM, CLASSIFIER_USER
from nova.errors import LLMUnavailableError, is_connection_error
from nova.observability import log_event
from nova.tools.calculator import extract_pure_expression

Period = Literal["5d", "1mo", "3mo", "6mo", "1y"]
FinanceNeed = Literal["quote", "history", "news"]
InsightKind = Literal["summary", "decisions", "action_items", "open_questions"]


class QueryClassification(BaseModel):
    query_type: Literal[
        "general", "web", "finance", "document", "video", "calculation", "multi_tool", "clarification_required"
    ]
    needs_live_data: bool = False
    tickers: list[str] = Field(default_factory=list)
    finance_needs: list[FinanceNeed] = Field(default_factory=list)
    period: Period | None = None
    document_query: str | None = None
    media_query: str | None = None
    insight_request: list[InsightKind] = Field(default_factory=list)
    search_query: str | None = None
    expression: str | None = None
    needs_calculation: bool = False
    clarification_question: str | None = None

    @field_validator("tickers", mode="before")
    @classmethod
    def _clean_tickers(cls, value: Any) -> list[str]:
        if not value:
            return []
        if isinstance(value, str):
            value = re.split(r"[,\s]+", value)
        out = []
        for t in value:
            t = str(t).strip().upper().lstrip("$")
            if re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,9}", t) and t not in out:
                out.append(t)
        return out[:3]


_GREETING = re.compile(
    r"^(hi|hello|hey|yo|thanks|thank you|good (morning|afternoon|evening)|who are you|what can you do)\b[\s!.?]*$",
    re.I,
)
_TIME_SENSITIVE = re.compile(
    r"\b(today|tonight|current(ly)?|latest|right now|this (week|month|year)|recent(ly)?|news|headlines?|stock price|share price|trading at|market cap)\b",
    re.I,
)
_DOC_REF = re.compile(
    r"\b(my|the|this|uploaded|attached)\s+(?:[\w-]+\s+){0,2}?(pdf|document|doc|report|file|paper|resume|cv|deck|presentation)\b"
    r"|\bin the (pdf|document|report)\b|\baccording to\b",
    re.I,
)
_FINANCE_WORDS = re.compile(
    r"\b(stock|share|ticker|price|quote|volatility|drawdown|returns?|moving average|market|nasdaq|nyse|s&p|performance)\b",
    re.I,
)
_DERIVED = re.compile(
    r"\b(percent(age)?|what share|ratio|difference|how much (more|less|higher|lower)|growth( rate)?|grew|increase(d)?|"
    r"decrease(d)?|change(d)? (from|between)|how many years|times (larger|bigger|more)|per (employee|share|unit)|"
    r"multipl(y|ied)|divided?|sum of|total of|average of|compare|calculate|compute|total (cost|value)|"
    r"would [\w\s]{0,30}cost|\d+ shares)\b",
    re.I,
)
_MEDIA_REF = re.compile(
    r"\b(meeting|video|recording|call|transcript|audio|podcast|webinar|lecture|interview|clip|episode|standup|stand-up)s?\b"
    r"|\b(they|we|he|she|someone|anyone|the team)\s+(said|say|decided|decide|agreed|agree|discussed|discuss|mentioned|promised|talked)\b"
    r"|\b(discussed|decided|agreed|mentioned|said)\s+(in|during|on|at)\b|\bwho (said|mentioned|raised|proposed)\b",
    re.I,
)
_INSIGHT_PATTERNS: list[tuple[InsightKind, re.Pattern[str]]] = [
    ("summary", re.compile(r"\b(summar(y|ise|ize|ized|ised)|recap|overview|gist|tl;?dr)\b", re.I)),
    ("decisions", re.compile(r"\b(key )?decisions?\b|\bwhat (was|were) (decided|agreed)\b", re.I)),
    ("action_items", re.compile(r"\baction items?\b|\bto-?dos?\b|\bnext steps\b|\bfollow-?ups?\b|\btasks? (were )?assigned\b", re.I)),
    ("open_questions", re.compile(r"\bopen questions?\b|\bunresolved\b|\bunanswered\b|\boutstanding (questions|issues)\b", re.I)),
]


def insight_kinds(query: str) -> list[InsightKind]:
    """Meeting-intelligence sections the user explicitly asks for (summary, decisions, ...)."""
    return [kind for kind, pattern in _INSIGHT_PATTERNS if pattern.search(query)]


_NEEDS_TICKER = re.compile(r"\b(stock|share) price\b|\bticker\b|\bthe stock\b", re.I)
_NEWS = re.compile(r"\b(news|headlines?|announce|announcement)\b", re.I)
_HISTORY = re.compile(
    r"\b(performance|perform|returns?|volatility|drawdown|trend|history|historical|moving average|over the (last|past)|past (week|month|year)|last (week|month|year|\d+ (days|months)))\b",
    re.I,
)
_UPPER_TICKER = re.compile(r"(?<![A-Za-z$])\$?([A-Z]{2,5})(?![A-Za-z])")
_NOT_TICKERS = {"PDF", "CEO", "CFO", "USD", "EUR", "API", "AI", "US", "UK", "EU", "GDP", "ETF", "IPO", "FY", "Q1", "Q2", "Q3", "Q4", "EPS", "OK", "I"}
_COMMON_TICKERS = {
    "apple": "AAPL", "tesla": "TSLA", "microsoft": "MSFT", "nvidia": "NVDA", "amazon": "AMZN",
    "google": "GOOGL", "alphabet": "GOOGL", "meta": "META", "facebook": "META", "netflix": "NFLX",
    "amd": "AMD", "intel": "INTC", "ibm": "IBM", "oracle": "ORCL", "salesforce": "CRM",
}
_PERIOD_WORDS = [
    (re.compile(r"\b(week|5 ?d(ays)?)\b", re.I), "5d"),
    (re.compile(r"\b(3 ?months?|quarter)\b", re.I), "3mo"),
    (re.compile(r"\b(6 ?months?|half[- ]year)\b", re.I), "6mo"),
    (re.compile(r"\b(year|12 ?months?|1 ?y)\b", re.I), "1y"),
    (re.compile(r"\b(month|30 ?days)\b", re.I), "1mo"),
]


def guess_tickers(text: str) -> list[str]:
    found = [m.group(1) for m in _UPPER_TICKER.finditer(text) if m.group(1) not in _NOT_TICKERS]
    lowered = text.lower()
    found += [sym for name, sym in _COMMON_TICKERS.items() if re.search(rf"\b{name}\b", lowered)]
    return list(dict.fromkeys(found))[:3]


def guess_period(text: str) -> str | None:
    for pattern, period in _PERIOD_WORDS:
        if pattern.search(text):
            return period
    return None


def heuristic_classify(query: str, has_documents: bool, has_media: bool = False) -> QueryClassification:
    """Deterministic classifier — used for fast paths and as a fallback."""
    expr = extract_pure_expression(query)
    if expr:
        return QueryClassification(query_type="calculation", expression=expr)
    if _GREETING.match(query.strip()):
        return QueryClassification(query_type="general")

    tickers = guess_tickers(query)
    if not tickers and _NEEDS_TICKER.search(query) and not _DOC_REF.search(query):
        return QueryClassification(
            query_type="clarification_required",
            clarification_question="Which company or ticker symbol should I look up?",
        )
    wants_doc = bool(_DOC_REF.search(query)) and has_documents
    wants_video = has_media and bool(_MEDIA_REF.search(query) or (insight_kinds(query) and not wants_doc))
    wants_fin = bool(tickers) and bool(_FINANCE_WORDS.search(query) or _NEWS.search(query))
    # "current revenue is 8 crore" in a question about a meeting/document is a user-supplied
    # figure, not a request for live data.
    wants_web = bool(_TIME_SENSITIVE.search(query)) and not wants_fin and not (wants_video or wants_doc)
    needs = []
    if wants_fin:
        needs.append("history" if _HISTORY.search(query) else "quote")
        if _NEWS.search(query):
            needs.append("news")
    kinds = [k for k, on in (("document", wants_doc), ("video", wants_video), ("finance", wants_fin),
                             ("web", wants_web)) if on]
    qtype = "multi_tool" if len(kinds) > 1 else (kinds[0] if kinds else "general")
    return QueryClassification(
        query_type=qtype,  # type: ignore[arg-type]
        needs_live_data=wants_fin or wants_web,
        tickers=tickers,
        finance_needs=needs,  # type: ignore[arg-type]
        period=guess_period(query) if "history" in needs else None,  # type: ignore[arg-type]
        document_query=query if wants_doc else None,
        media_query=query if wants_video else None,
        insight_request=insight_kinds(query) if wants_video else [],
        search_query=query if wants_web else None,
    )


def apply_policy_guards(
    c: QueryClassification, query: str, has_documents: bool, has_media: bool = False
) -> tuple[QueryClassification, list[str]]:
    """Deterministic overrides of the model's routing. Returns (classification, notes)."""
    notes: list[str] = []
    c = c.model_copy(deep=True)

    if not c.tickers and c.query_type in ("finance", "multi_tool"):
        c.tickers = guess_tickers(query)

    grounded_ref = (has_media and _MEDIA_REF.search(query)) or (has_documents and _DOC_REF.search(query))
    if c.query_type == "general" and _TIME_SENSITIVE.search(query) and not grounded_ref:
        if c.tickers and _FINANCE_WORDS.search(query):
            c.query_type = "finance"
        else:
            c.query_type = "web"
            c.search_query = c.search_query or query
        c.needs_live_data = True
        notes.append("time-sensitive wording: routed to live data")

    if has_documents and _DOC_REF.search(query) and c.query_type in ("general", "web"):
        c.query_type = "document" if c.query_type == "general" else "multi_tool"
        notes.append("explicit document reference: routed to document retrieval")

    if has_media and _MEDIA_REF.search(query) and c.query_type in ("general", "web", "document"):
        c.query_type = "video" if c.query_type == "general" else "multi_tool"
        notes.append("explicit media reference: routed to video retrieval")

    if c.query_type == "video" and not has_media and not _MEDIA_REF.search(query):
        c.query_type = "general"  # the model guessed video without any media reference
        notes.append("video route dropped: no media and no media reference")
    if c.query_type in ("video", "multi_tool") and has_media and not c.media_query and (
        c.query_type == "video" or _MEDIA_REF.search(query)
    ):
        c.media_query = query
    if c.media_query and not c.insight_request:
        c.insight_request = insight_kinds(query)

    if c.query_type in ("document", "multi_tool") and has_documents and not c.document_query and (
        c.query_type == "document" or _DOC_REF.search(query)
    ):
        c.document_query = query

    if c.query_type == "finance":
        if not c.tickers:
            c.query_type = "clarification_required"
            c.clarification_question = c.clarification_question or (
                "Which company or ticker symbol should I look up?"
            )
            notes.append("finance request without ticker")
        elif not c.finance_needs:
            c.finance_needs = ["history"] if _HISTORY.search(query) else ["quote"]
        if "history" in c.finance_needs and not c.period:
            c.period = guess_period(query) or "1mo"  # type: ignore[assignment]
        if _NEWS.search(query) and "news" not in c.finance_needs:
            c.finance_needs.append("news")

    if c.query_type == "calculation" and not c.expression:
        c.expression = extract_pure_expression(query)

    # Derived figures (percent of total, growth, differences) must come from the
    # calculator, never from the model's mental arithmetic.
    if c.query_type in ("document", "video", "web", "finance", "multi_tool") and not c.expression and _DERIVED.search(query):
        c.needs_calculation = True

    if c.query_type == "web" and not c.search_query:
        c.search_query = query

    if c.query_type == "clarification_required" and not c.clarification_question:
        c.clarification_question = "Could you add a bit more detail about what you need?"
    return c, notes


_FOLLOW_UP = re.compile(r"\b(it|its|they|them|their|that|those|this|these|he|she|his|her|earlier|previous|above)\b", re.I)


def is_confident(c: QueryClassification, query: str, history: str, has_documents: bool, has_media: bool = False) -> bool:
    """
    High-precision rule matches that make an LLM routing call unnecessary. Anything
    ambiguous — follow-ups needing rewriting, mixed signals, uploaded documents that
    might be relevant — still goes to the LLM.
    """
    if c.query_type == "calculation" and c.expression:
        return True
    if c.query_type == "general":
        return bool(_GREETING.match(query.strip()))
    if c.query_type == "clarification_required":
        return not history  # with history, the ticker may come from an earlier turn
    if history and _FOLLOW_UP.search(query):
        return False
    if c.query_type == "finance":
        return bool(c.tickers) and not has_documents and not has_media
    if c.query_type == "video":
        return not _FINANCE_WORDS.search(query) and not _TIME_SENSITIVE.search(query)
    if c.query_type == "document":
        return not _FINANCE_WORDS.search(query) and not _TIME_SENSITIVE.search(query)
    if c.query_type == "web":
        return not has_documents and not has_media
    return False


def classify(
    llm: BaseChatModel,
    query: str,
    history: str,
    documents: list[str],
    media: list[str] | None = None,
) -> tuple[QueryClassification, str, list[str]]:
    """Return (classification, method, notes). method ∈ {rule, llm, fallback}."""
    has_docs, has_media = bool(documents), bool(media)
    fast = heuristic_classify(query, has_docs, has_media)
    if is_confident(fast, query, history, has_docs, has_media):
        guarded, notes = apply_policy_guards(fast, query, has_docs, has_media)
        return guarded, "rule", notes

    try:
        structured = llm.with_structured_output(QueryClassification, method="json_schema")
        result = structured.invoke(
            [
                SystemMessage(content=CLASSIFIER_SYSTEM),
                HumanMessage(
                    content=CLASSIFIER_USER.format(
                        documents=", ".join(documents) if documents else "none",
                        media=", ".join(media or []) or "none",
                        history=history or "(none)",
                        query=query,
                    )
                ),
            ]
        )
        if not isinstance(result, QueryClassification):
            result = QueryClassification.model_validate(result)
        method = "llm"
    except Exception as exc:  # noqa: BLE001 - LLM/parse failures fall back to rules
        if is_connection_error(exc):
            raise LLMUnavailableError(repr(exc)) from exc
        log_event("classifier_fallback", level=30, error=repr(exc))
        result, method = fast, "fallback"

    guarded, notes = apply_policy_guards(result, query, has_docs, has_media)
    return guarded, method, notes
