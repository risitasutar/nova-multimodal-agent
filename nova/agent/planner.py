"""
Planner: turns a classification into an ordered, bounded list of tool steps.

Single-capability routes get a deterministic plan derived from the classifier's
entities (no extra LLM call). Only `multi_tool` requests use the structured LLM
planner; its output is validated (allowed tools, no duplicates, step budget,
document steps only when a document exists) and falls back to a deterministic plan.
"""

from __future__ import annotations

import json
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from nova.agent.classifier import QueryClassification
from nova.agent.prompts import PLANNER_SYSTEM, PLANNER_USER
from nova.errors import LLMUnavailableError, is_connection_error
from nova.observability import log_event
from nova.tools.registry import TOOL_SPECS

ToolName = Literal["document_search", "video_search", "media_insights", "web_search", "news_search", "stock_quote",
                   "price_history", "calculator"]
_MEDIA_TOOLS = {"video_search", "media_insights"}


class PlanStep(BaseModel):
    tool: ToolName
    input: str = Field(description="search query, ticker symbol or arithmetic expression")
    period: Literal["5d", "1mo", "3mo", "6mo", "1y"] | None = None
    purpose: str = ""


class Plan(BaseModel):
    steps: list[PlanStep] = Field(default_factory=list)
    needs_calculation: bool = False
    summary: str = ""


def deterministic_plan(c: QueryClassification, query: str, has_documents: bool, has_media: bool = False) -> Plan:
    steps: list[PlanStep] = []
    if c.media_query and has_media:
        if c.insight_request:
            steps.append(PlanStep(tool="media_insights", input=",".join(c.insight_request), purpose="Meeting insights"))
        steps.append(PlanStep(tool="video_search", input=c.media_query, purpose="Meeting evidence"))
    if c.document_query and has_documents:
        steps.append(PlanStep(tool="document_search", input=c.document_query, purpose="Document retrieval"))
    for ticker in c.tickers:
        needs = c.finance_needs or ["quote"]
        if "quote" in needs:
            steps.append(PlanStep(tool="stock_quote", input=ticker, purpose=f"Live quote for {ticker}"))
        if "history" in needs:
            steps.append(
                PlanStep(tool="price_history", input=ticker, period=c.period or "1mo", purpose=f"{ticker} price history")
            )
        if "news" in needs:
            steps.append(PlanStep(tool="news_search", input=f"{ticker} stock news", purpose=f"{ticker} news"))
    if c.query_type in ("web", "multi_tool") and c.search_query and not any(
        s.tool == "news_search" for s in steps
    ):
        steps.append(PlanStep(tool="web_search", input=c.search_query, purpose="Web research"))
    if c.expression:
        steps.append(PlanStep(tool="calculator", input=c.expression, purpose="Calculation"))
    needs_calc = (c.needs_calculation or c.query_type == "multi_tool") and not c.expression
    return Plan(steps=steps, needs_calculation=needs_calc, summary="")


def validate_plan(plan: Plan, has_documents: bool, max_steps: int, has_media: bool = False) -> tuple[Plan, list[str]]:
    notes: list[str] = []
    seen: set[tuple[str, str, str | None]] = set()
    steps: list[PlanStep] = []
    for step in plan.steps:
        if step.tool == "document_search" and not has_documents:
            notes.append("dropped document_search: no document uploaded")
            continue
        if step.tool in _MEDIA_TOOLS and not has_media:
            notes.append(f"dropped {step.tool}: no media in this conversation")
            continue
        if step.tool in ("stock_quote", "price_history"):
            step.input = step.input.strip().upper().lstrip("$")
        if step.tool == "price_history" and not step.period:
            step.period = "1mo"
        key = (step.tool, step.input.strip().lower(), step.period)
        if not step.input.strip() or key in seen:
            continue
        seen.add(key)
        steps.append(step)
    if len(steps) > max_steps:
        notes.append(f"plan truncated to {max_steps} steps")
        steps = steps[:max_steps]
    return Plan(steps=steps, needs_calculation=plan.needs_calculation, summary=plan.summary), notes


def plan_query(
    llm: BaseChatModel,
    c: QueryClassification,
    query: str,
    documents: list[str],
    max_steps: int,
    media: list[str] | None = None,
) -> tuple[Plan, str, list[str]]:
    """Return (plan, method, notes). method ∈ {deterministic, llm, fallback}."""
    has_docs, has_media = bool(documents), bool(media)
    if c.query_type != "multi_tool":
        plan, notes = validate_plan(deterministic_plan(c, query, has_docs, has_media), has_docs, max_steps, has_media)
        return plan, "deterministic", notes

    method = "llm"
    try:
        structured = llm.with_structured_output(Plan, method="json_schema")
        raw = structured.invoke(
            [
                SystemMessage(content=PLANNER_SYSTEM.format(max_steps=max_steps)),
                HumanMessage(
                    content=PLANNER_USER.format(
                        documents=", ".join(documents) if documents else "none",
                        media=", ".join(media or []) or "none",
                        classification=json.dumps(c.model_dump(exclude_none=True)),
                        query=query,
                    )
                ),
            ]
        )
        plan = raw if isinstance(raw, Plan) else Plan.model_validate(raw)
    except Exception as exc:  # noqa: BLE001 - planner failure falls back to deterministic plan
        if is_connection_error(exc):
            raise LLMUnavailableError(repr(exc)) from exc
        log_event("planner_fallback", level=30, error=repr(exc))
        plan, method = deterministic_plan(c, query, has_docs, has_media), "fallback"

    plan, notes = validate_plan(plan, has_docs, max_steps, has_media)
    if not plan.steps:  # the model produced nothing usable
        plan, notes2 = validate_plan(deterministic_plan(c, query, has_docs, has_media), has_docs, max_steps, has_media)
        method, notes = "fallback", notes + notes2
    # The router's evidence requirements are binding: a model plan cannot silently skip a
    # source the user explicitly referred to (e.g. "compare the meeting with the PDF").
    required = [st for st in deterministic_plan(c, query, has_docs, has_media).steps
                if st.tool in ("document_search", "video_search", "media_insights")]
    for st in required:
        if not any(x.tool == st.tool for x in plan.steps):
            plan.steps.insert(0, st)
            notes.append(f"added required {st.tool}")
    if c.expression and not any(s.tool == "calculator" for s in plan.steps):
        plan.steps.append(PlanStep(tool="calculator", input=c.expression, purpose="Calculation"))
    plan.needs_calculation = plan.needs_calculation or (c.needs_calculation and not c.expression)
    return plan, method, notes


def plan_labels(steps: list[dict]) -> list[str]:
    return [TOOL_SPECS[s["tool"]].label for s in steps if s.get("tool") in TOOL_SPECS]
