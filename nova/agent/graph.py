"""
Nova's LangGraph agent.

    START → validate_input ─┬─(invalid)──────────────────────────────────────────► finalize_response
                            └─► classify_query ─┬─ general ───────────────────────► generate_response
                                                ├─ clarification ─────────────────► finalize_response
                                                └─ tools ─► planner ─┬─► approval_gate (interrupt) ─┐
                                                                     └──────────────────────────────┴► execute_tools
    execute_tools ─► compute ─► check_evidence ─┬─ insufficient ─► finalize_response
                                                └─► generate_response ─► verify_response ─┬─ RETRY ─► generate_response
                                                                                          └─► finalize_response → END
All external effects go through `AgentDeps`, so tests inject fakes without patching.
"""

from __future__ import annotations

import concurrent.futures
import functools
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from nova.agent import prompts
from nova.agent.classifier import _GREETING, classify
from nova.agent.planner import plan_labels, plan_query
from nova.agent.state import PER_TURN_DEFAULTS, AgentState
from nova.agent.verifier import (
    check_claim_support,
    check_numbers,
    extract_numbers,
    is_abstention,
    number_supported,
)
from nova.config import Settings, get_settings
from nova.errors import InputValidationError, LLMUnavailableError, is_connection_error
from nova.observability import log_event, metrics
from nova.rag.citations import apply_citations, build_sources, citation_record, format_sources_for_prompt
from nova.rag.retrieve import Retriever, tokenize
from nova.rag.store import DocumentStore
from nova.safety import clean_user_message
from nova.tools import calculator as calc_mod
from nova.tools.base import ToolResult
from nova.tools.finance import price_history, stock_quote
from nova.tools.registry import TOOL_SPECS
from nova.tools.web import news_search, web_search

if TYPE_CHECKING:
    from nova.video.service import MediaService

_THINK = re.compile(r"<think>.*?</think>", re.S)
_MEDIA_TOOLS = {"video_search", "media_insights"}
_EVIDENCE_TOOLS = {"document_search", "calculator", *_MEDIA_TOOLS}
_INSIGHT_LABEL = {"summary": "Summary point", "decisions": "Key decision", "action_items": "Action item",
                  "open_questions": "Open question"}
_GENERIC_NAME_TOKENS = {"mp4", "mov", "avi", "mkv", "webm", "mp3", "wav", "m4a", "srt", "vtt", "video", "audio",
                        "meeting", "recording", "call", "youtube", "the", "and", "final", "copy"}
_ALL_MEDIA = re.compile(r"\b(both|all|each|every|across|between|meetings|videos|recordings|calls)\b", re.I)
_NAME_TOKEN = re.compile(r"[a-z0-9]{3,}")


@dataclass
class AgentDeps:
    """Everything the graph touches outside itself."""

    llm: BaseChatModel
    store: DocumentStore
    retriever: Retriever
    settings: Settings = field(default_factory=get_settings)
    tools: dict[str, Callable[..., ToolResult]] = field(
        default_factory=lambda: {
            "web_search": web_search,
            "news_search": news_search,
            "stock_quote": stock_quote,
            "price_history": price_history,
            "calculator": calc_mod.calculate,
        }
    )
    media: MediaService | None = None  # video/audio capability (None = disabled)


class _Calculation(BaseModel):
    label: str
    expression: str


class _CalculationPlan(BaseModel):
    calculations: list[_Calculation] = Field(default_factory=list)


# ----------------------------------------------------------------- helpers
def _thread_id(config: RunnableConfig) -> str:
    return str((config.get("configurable") or {}).get("thread_id", ""))


def _approval_required(config: RunnableConfig, settings: Settings) -> bool:
    value = (config.get("configurable") or {}).get("approval_required")
    return settings.approval_required_default if value is None else bool(value)


def conversation_history(messages: list[BaseMessage], turns: int, max_chars: int = 600) -> list[BaseMessage]:
    """Prior user/assistant turns as plain, truncated text (no tool plumbing, no metadata)."""
    clean: list[BaseMessage] = []
    for m in messages[:-1]:
        if isinstance(m, HumanMessage) and isinstance(m.content, str):
            clean.append(HumanMessage(content=m.content[:max_chars]))
        elif isinstance(m, AIMessage) and m.content and not m.tool_calls:
            clean.append(AIMessage(content=str(m.content)[:max_chars]))
    return clean[-turns * 2 :] if turns > 0 else []


def history_text(messages: list[BaseMessage], max_chars: int = 1500) -> str:
    lines = [
        f"{'User' if isinstance(m, HumanMessage) else 'Nova'}: {str(m.content)[:300]}" for m in messages[-6:]
    ]
    return "\n".join(lines)[-max_chars:]


def timed(name: str) -> Callable:
    """Record node wall-time into latency_metadata and the metrics registry."""

    def deco(fn: Callable[..., dict]) -> Callable[..., dict]:
        @functools.wraps(fn)
        def wrapper(state: AgentState, config: RunnableConfig) -> dict:
            start = time.perf_counter()
            update = fn(state, config) or {}
            ms = round((time.perf_counter() - start) * 1000, 1)
            metrics.observe(f"node.{name}", ms)
            lat = dict(update.get("latency_metadata") or {})
            # Nodes can run more than once per turn (verification retry): accumulate.
            prior = 0.0 if lat.get("_reset") else (state.get("latency_metadata") or {}).get(name, 0.0)
            lat[name] = round(prior + ms, 1)
            update["latency_metadata"] = lat
            return update

        return wrapper

    return deco


def _llm_call(fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except LLMUnavailableError:
        raise
    except Exception as exc:
        if is_connection_error(exc):
            raise LLMUnavailableError(repr(exc)) from exc
        raise


def _fmt_num(value: Any) -> str:
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.6f}".rstrip("0").rstrip(".")
    return str(value)


def finance_evidence(result: dict[str, Any]) -> dict[str, str] | None:
    """Turn a finance ToolResult into a compact, citable text source."""
    d = result["data"]
    if result["tool"] == "stock_quote":
        parts = [f"{d['symbol']}" + (f" ({d['name']})" if d.get("name") else "") + f" latest price as of {d['as_of']}: {d['price']} {d.get('currency') or ''}".rstrip()]
        if d.get("previous_close") is not None:
            parts.append(f"previous close {d['previous_close']}")
        if d.get("change") is not None:
            parts.append(f"change {d['change']:+} ({d['change_percent']:+}%)")
        if d.get("day_low") is not None and d.get("day_high") is not None:
            parts.append(f"day range {d['day_low']}–{d['day_high']}")
        if d.get("fifty_two_week_low") is not None:
            parts.append(f"52-week range {d['fifty_two_week_low']}–{d['fifty_two_week_high']}")
        return {"label": result["source"] or d["symbol"], "text": "; ".join(parts) + ".", "symbol": d["symbol"],
                "metric": "quote", "provider": d.get("provider"), "retrieved_at": result.get("timestamp")}
    if result["tool"] == "price_history":
        a = d["analytics"]
        text = (
            f"{d['symbol']} daily closing prices {a['start_date']} to {a['end_date']} ({a['trading_days']} trading days, {d.get('currency') or ''}): "
            f"start {a['start_price']}, end {a['end_price']}, period return {a['period_return_pct']}%, "
            f"period high {a['period_high']}, period low {a['period_low']}, "
            f"annualized volatility {a['annualized_volatility_pct']}%, max drawdown {a['max_drawdown_pct']}%"
        )
        if a.get("sma_20") is not None:
            text += f", 20-day SMA {a['sma_20']}"
        if a.get("sma_50") is not None:
            text += f", 50-day SMA {a['sma_50']}"
        if a.get("best_day"):
            text += f", best day {a['best_day']['date']} ({a['best_day']['return_pct']}%), worst day {a['worst_day']['date']} ({a['worst_day']['return_pct']}%)"
        return {"label": result["source"] or d["symbol"], "text": text + ".", "symbol": d["symbol"],
                "metric": f"price_history:{d.get('period')}", "provider": d.get("provider"),
                "retrieved_at": result.get("timestamp")}
    return None


def select_media(query: str, assets: list[Any]) -> tuple[list[str], bool]:
    """
    Pick which recordings a question refers to. Returns (media_ids, ambiguous).
    A filename match wins; "both/all/compare the meetings" means all; a singular generic
    reference ("the meeting") with several recordings and no name is ambiguous.
    """
    if not assets:
        return [], False
    q_tokens = set(_NAME_TOKEN.findall(query.lower()))
    named = [a.media_id for a in assets
             if (set(_NAME_TOKEN.findall(a.filename.lower())) - _GENERIC_NAME_TOKENS) & q_tokens]
    if named:
        return named, False
    if len(assets) == 1 or _ALL_MEDIA.search(query):
        return [a.media_id for a in assets], False
    return [a.media_id for a in assets], True


def _insight_chunks(media: Any, tid: str, media_ids: list[str], kinds: list[str]) -> tuple[list[dict], list[str], list[str]]:
    """Stored meeting insights → timestamped evidence chunks. Returns (chunks, missing, empty_sections)."""
    chunks: list[dict[str, Any]] = []
    missing: list[str] = []
    empty: list[str] = []
    for mid in media_ids:
        asset = media.get(mid, tid)
        insights = media.insights(mid, tid)
        if insights is None:
            missing.append(asset.filename)
            continue
        for kind in kinds:
            items = getattr(insights, kind)
            if not items:
                empty.append(f"{_INSIGHT_LABEL[kind]}s: none identified in the transcript of {asset.filename}.")
            for n, item in enumerate(items):
                text = f"{_INSIGHT_LABEL[kind]}: {item.text}"
                if kind == "action_items":
                    text += f" | Owner: {item.owner or 'Not identified in the transcript.'}"
                    text += f" | Deadline: {item.deadline or 'Not identified in the transcript.'}"
                chunks.append({
                    "chunk_id": f"{mid[:8]}-{kind}-{n}", "media_id": mid, "text": text,
                    "start_time": item.start_time or 0.0, "end_time": item.end_time or 0.0,
                    "source_type": asset.media_kind, "source_name": asset.filename,
                    "duration_seconds": asset.duration_seconds, "insight": kind,
                })
    return chunks, missing, empty


def _run_with_timeout(fn: Callable[..., ToolResult], timeout_s: float, name: str, **kwargs: Any) -> ToolResult:
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = pool.submit(fn, **kwargs)
    try:
        return future.result(timeout=timeout_s)
    except concurrent.futures.TimeoutError:
        log_event("tool_timeout", level=30, tool=name, timeout_s=timeout_s)
        metrics.incr(f"tool.{name}.timeout")
        return ToolResult(tool=name, status="error", error=f"{name} timed out.", input=kwargs)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def build_timeline(state: dict[str, Any]) -> list[str]:
    """User-facing execution summary. Describes actions, never model reasoning."""
    tl = [f"✓ Query analyzed → {state.get('query_type', 'general')}"]
    if state.get("plan"):
        tl.append("✓ " + state.get("plan_summary", "Plan created"))
    approval = state.get("approval")
    if approval:
        tl.append("✓ External data approved" if approval.get("approved") else "✗ External data rejected by user")
    for r in state.get("tool_results", []):
        if r["tool"] == "document_search":
            n = r["data"].get("chunks", 0)
            tl.append(f"{'✓' if n else '⚠'} {n} document chunk{'s' if n != 1 else ''} retrieved")
        elif r["tool"] == "video_search" and r["status"] == "success":
            n = r["data"].get("chunks", 0)
            tl.append(f"✓ Retrieved meeting evidence ({n} transcript chunk{'s' if n != 1 else ''})" if n
                      else "⚠ No relevant meeting passage found")
        elif r["tool"] == "media_insights" and r["status"] == "success":
            tl.append(f"✓ Meeting insights loaded ({', '.join(r['data'].get('sections', []))})")
        elif r["status"] == "success":
            label = {
                "web_search": f"{len(r['data'].get('results', []))} web sources found",
                "news_search": f"{len(r['data'].get('results', []))} news articles found",
                "stock_quote": f"Market quote retrieved ({r['data'].get('symbol')})",
                "price_history": f"Price history analysed ({r['data'].get('symbol')}, {r['data'].get('period')})",
                "calculator": f"Calculation completed ({r['data'].get('expression')})",
            }.get(r["tool"], f"{r['tool']} completed")
            tl.append("✓ " + label)
        else:
            tl.append(f"✗ {r['tool']} failed: {r.get('error')}")
    status = state.get("verification_status")
    if status == "VERIFIED":
        tl.append("✓ Evidence verified")
    elif status == "INSUFFICIENT_EVIDENCE":
        tl.append("⚠ Insufficient evidence")
    elif status == "FAILED":
        tl.append("⚠ Verification incomplete")
    return tl


# ----------------------------------------------------------------- graph
def build_graph(deps: AgentDeps) -> StateGraph:
    s = deps.settings

    # ---------- validate_input ----------
    @timed("validate_input")
    def validate_input(state: AgentState, config: RunnableConfig) -> dict:
        last = state["messages"][-1] if state.get("messages") else None
        reset = {**PER_TURN_DEFAULTS, "latency_metadata": {"_reset": 1}}
        if not isinstance(last, HumanMessage):
            return reset | {"error_state": {"code": "invalid_input"}, "final_response": "Please enter a message.",
                            "verification_status": "SKIPPED"}
        try:
            text = clean_user_message(str(last.content))
        except InputValidationError as exc:
            return reset | {"error_state": exc.to_dict(), "final_response": exc.user_message,
                            "verification_status": "SKIPPED"}
        return reset | {"user_query": text}

    # ---------- classify_query ----------
    @timed("classify_query")
    def classify_query(state: AgentState, config: RunnableConfig) -> dict:
        tid = _thread_id(config)
        docs = [d["filename"] for d in deps.store.list_documents(tid)] if tid else []
        assets = deps.media.completed(tid) if (deps.media and tid) else []
        media_names = [a.filename for a in assets]
        query = state["user_query"]
        hist = conversation_history(state["messages"], s.history_turns, s.history_message_chars)
        t0 = time.perf_counter()
        c, method, notes = classify(deps.llm, query, history_text(hist), docs, media_names)
        llm_ms = round((time.perf_counter() - t0) * 1000, 1)

        # Retrieval-aware routing for media: a strong transcript match pulls in video evidence.
        if assets and c.query_type in ("general", "web", "document") and not _GREETING.match(query) and deps.media:
            probe = deps.media.search(tid, query, min_relevance=s.video_min_relevance + 0.05, top_k=1)
            if probe.sufficient:
                c.media_query = c.media_query or query
                c.query_type = "video" if c.query_type == "general" else "multi_tool"
                notes.append(f"media probe matched (relevance {probe.max_relevance:.2f})")

        # Retrieval-aware routing: if the model chose general/web but the uploaded
        # document clearly covers the question, include document retrieval.
        if docs and c.query_type in ("general", "web") and not _GREETING.match(query):
            probe = deps.retriever.retrieve(tid, query, min_relevance=s.min_relevance + 0.05, top_k=1)
            if probe.sufficient:
                c.document_query = c.document_query or query
                c.query_type = "document" if c.query_type == "general" else "multi_tool"
                notes.append(f"document probe matched (relevance {probe.max_relevance:.2f})")

        media_ids: list[str] = []
        update_active: dict[str, Any] = {}
        if c.media_query and assets:
            media_ids, ambiguous = select_media(query, assets)
            active = [m for m in (state.get("active_media") or []) if m in {a.media_id for a in assets}]
            if ambiguous and len(active) == 1:
                media_ids, ambiguous = active, False  # memory: the recording discussed in earlier turns
                notes.append("media resolved from conversation context")
            if ambiguous:
                c.query_type = "clarification_required"
                c.clarification_question = ("This conversation has several recordings: "
                                            + ", ".join(f"**{n}**" for n in media_names)
                                            + ". Which one do you mean (or say 'all of them')?")
                notes.append("ambiguous media reference")
            else:
                update_active = {"active_media": media_ids}

        log_event("query_classified", route=c.query_type, method=method, notes=notes)
        return {
            "query_type": c.query_type,
            "classification": c.model_dump() | {"method": method, "notes": notes, "documents": docs,
                                                "media": media_names, "media_ids": media_ids},
            "latency_metadata": {"llm.classify": llm_ms if method != "rule" else 0.0},
            **update_active,
        }

    def route_after_classify(state: AgentState) -> str:
        qt = state["query_type"]
        if qt == "general":
            return "generate_response"
        if qt == "clarification_required":
            return "finalize_response"
        return "planner"

    # ---------- planner ----------
    @timed("planner")
    def planner(state: AgentState, config: RunnableConfig) -> dict:
        from nova.agent.classifier import QueryClassification

        cls = state["classification"]
        c = QueryClassification.model_validate({k: v for k, v in cls.items() if k in QueryClassification.model_fields})
        docs = cls.get("documents", [])
        t0 = time.perf_counter()
        plan, method, notes = plan_query(deps.llm, c, state["user_query"], docs, s.max_plan_steps,
                                         cls.get("media", []))
        steps = [st.model_dump() for st in plan.steps]
        update: dict[str, Any] = {
            "plan": steps,
            "needs_calculation": plan.needs_calculation,
            "selected_tools": list(dict.fromkeys(st["tool"] for st in steps)),
            "plan_summary": " → ".join(["Plan created", *plan_labels(steps), "Verification"]),
            "latency_metadata": {"llm.plan": round((time.perf_counter() - t0) * 1000, 1) if method == "llm" else 0.0},
        }
        log_event("plan_created", method=method, steps=[(st["tool"], st["input"]) for st in steps], notes=notes)
        if not steps:
            if c.query_type == "video" or (c.media_query and not c.document_query):
                update["final_response"] = prompts.NO_MEDIA_UPLOADED
                update["verification_status"] = "INSUFFICIENT_EVIDENCE"
            elif c.query_type == "document" or c.document_query:
                update["final_response"] = prompts.NO_DOCUMENT_UPLOADED
                update["verification_status"] = "INSUFFICIENT_EVIDENCE"
            return update
        external = [st for st in steps if TOOL_SPECS[st["tool"]].external]
        update["requires_human_approval"] = bool(external) and _approval_required(config, s)
        return update

    def route_after_planner(state: AgentState) -> str:
        if state.get("final_response"):
            return "finalize_response"
        if not state.get("plan"):
            return "generate_response"
        return "approval_gate" if state.get("requires_human_approval") else "execute_tools"

    # ---------- approval_gate (human-in-the-loop) ----------
    @timed("approval_gate")
    def approval_gate(state: AgentState, config: RunnableConfig) -> dict:
        operations = [
            {"tool": st["tool"], "input": st["input"], "period": st.get("period"),
             "description": TOOL_SPECS[st["tool"]].description}
            for st in state["plan"] if TOOL_SPECS[st["tool"]].external
        ]
        decision = interrupt(
            {
                "type": "approval_required",
                "message": "Nova wants to execute the following external data operations.",
                "operations": operations,
            }
        )
        approved = bool(decision.get("approved")) if isinstance(decision, dict) else bool(decision)
        log_event("approval_decision", approved=approved, operations=len(operations))
        if approved:
            return {"approval": {"approved": True, "operations": operations}}
        remaining = [st for st in state["plan"] if not TOOL_SPECS[st["tool"]].external]
        update: dict[str, Any] = {
            "approval": {"approved": False, "operations": operations},
            "plan": remaining,
            "selected_tools": list(dict.fromkeys(st["tool"] for st in remaining)),
        }
        if not remaining:
            update["final_response"] = (
                "Understood — I did not run the external data operations, so I can't answer this "
                "with live data. Ask again and approve the request if you'd like me to proceed."
            )
            update["verification_status"] = "SKIPPED"
        return update

    def route_after_approval(state: AgentState) -> str:
        return "finalize_response" if state.get("final_response") else "execute_tools"

    # ---------- execute_tools ----------
    @timed("execute_tools")
    def execute_tools(state: AgentState, config: RunnableConfig) -> dict:
        tid = _thread_id(config)
        results: list[dict[str, Any]] = []
        chunks: dict[str, dict[str, Any]] = {}
        video: dict[str, dict[str, Any]] = {}
        insight_chunks: list[dict[str, Any]] = []
        media_ids = list(state.get("classification", {}).get("media_ids") or [])
        latency: dict[str, float] = {}
        ordered = sorted(state["plan"], key=lambda st: st["tool"] == "calculator")  # data first
        for i, step in enumerate(ordered, start=1):
            name, spec = step["tool"], TOOL_SPECS[step["tool"]]
            if name == "document_search":
                r = deps.retriever.retrieve(tid, state["user_query"], rewritten=step["input"])
                for ch in r.chunks:
                    chunks.setdefault(ch["chunk_id"], ch)
                tr = ToolResult(
                    tool=name, status="success", source="uploaded documents", latency_ms=r.latency_ms,
                    input={"query": step["input"]},
                    data={"queries": r.queries, "chunks": len(r.chunks), "candidates": r.candidates,
                          "max_relevance": round(r.max_relevance, 4)},
                )
                metrics.observe("retrieval", r.latency_ms)
            elif name in _MEDIA_TOOLS and deps.media is None:
                tr = ToolResult(tool=name, status="error", error="Video capability is not enabled.", input={})
            elif name == "video_search":
                assert deps.media is not None
                # thread_id comes from the run config and media ids from the router — never from tool input.
                r = deps.media.search(tid, state["user_query"], rewritten=step["input"], media_ids=media_ids or None)
                for ch in r.chunks:
                    video.setdefault(ch["chunk_id"], ch)
                tr = ToolResult(
                    tool=name, status="success", source="conversation media", latency_ms=r.latency_ms,
                    input={"query": step["input"], "media_ids": media_ids},
                    data={"queries": r.queries, "chunks": len(r.chunks), "candidates": r.candidates,
                          "max_relevance": round(r.max_relevance, 4),
                          "hits": [{"media_id": c["media_id"], "start_time": c["start_time"], "end_time": c["end_time"],
                                    "relevance": c["dense_score"]} for c in r.chunks]},
                )
            elif name == "media_insights":
                assert deps.media is not None
                kinds = [k for k in step["input"].replace(" ", "").split(",") if k in _INSIGHT_LABEL] or list(_INSIGHT_LABEL)
                ids = media_ids or [a.media_id for a in deps.media.completed(tid)]
                t_ins = time.perf_counter()
                found, missing, empty = _insight_chunks(deps.media, tid, ids, kinds)
                insight_chunks.extend(found)
                ok = bool(found or empty)
                tr = ToolResult(
                    tool=name, status="success" if ok else "error", source="conversation media",
                    latency_ms=round((time.perf_counter() - t_ins) * 1000, 1), input={"sections": kinds},
                    data={"sections": kinds, "items": len(found), "missing": missing, "empty_sections": empty},
                    error=None if ok else "Meeting insights have not been generated for: " + ", ".join(missing),
                )
            elif name == "calculator":
                tr = _run_with_timeout(deps.tools[name], spec.timeout_s, name, expression=step["input"])
            elif name == "price_history":
                tr = _run_with_timeout(deps.tools[name], spec.timeout_s, name,
                                       symbol=step["input"], period=step.get("period") or "1mo")
            elif name in ("stock_quote",):
                tr = _run_with_timeout(deps.tools[name], spec.timeout_s, name, symbol=step["input"])
            else:
                tr = _run_with_timeout(deps.tools[name], spec.timeout_s, name, query=step["input"])
            latency[f"tool.{name}#{i}"] = tr.latency_ms
            results.append(tr.model_dump())

        ranked = sorted(chunks.values(), key=lambda c: c["score"], reverse=True)[: s.retrieval_top_k]
        ranked_video = sorted(video.values(), key=lambda c: c["score"], reverse=True)[: s.retrieval_top_k]
        media_evidence = insight_chunks + ranked_video
        return {
            "tool_results": results,
            "retrieved_documents": ranked,
            "retrieved_media": ranked_video,
            "media_evidence": media_evidence,
            "sources": _sources_from(ranked, results, media_evidence),
            "retrieval_attempts": 1 if any(st["tool"] in ("document_search", "video_search") for st in ordered) else 0,
            "latency_metadata": latency,
        }

    def _sources_from(chunks: list[dict[str, Any]], results: list[dict[str, Any]],
                      media_evidence: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        web: list[dict[str, Any]] = []
        fin: list[dict[str, str]] = []
        for r in results:
            if r["status"] != "success":
                continue
            if r["tool"] in ("web_search", "news_search"):
                web.extend({**w, "retrieved_at": r.get("timestamp")} for w in r["data"].get("results", []))
            elif r["tool"] in ("stock_quote", "price_history"):
                ev = finance_evidence(r)
                if ev:
                    fin.append(ev)
        return build_sources(chunks, web, fin, media_evidence or [])

    # ---------- compute (calculator over gathered evidence) ----------
    @timed("compute")
    def compute(state: AgentState, config: RunnableConfig) -> dict:
        if not state.get("needs_calculation") or not state.get("sources"):
            return {}
        evidence = format_sources_for_prompt(state["sources"])
        t0 = time.perf_counter()
        try:
            raw = _llm_call(lambda: deps.llm.with_structured_output(_CalculationPlan, method="json_schema").invoke(
                [SystemMessage(content=prompts.COMPUTE_SYSTEM),
                 HumanMessage(content=f"{evidence}\n\nUser request: {state['user_query']}")]
            ))
            plan = raw if isinstance(raw, _CalculationPlan) else _CalculationPlan.model_validate(raw)
        except LLMUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001
            log_event("compute_plan_failed", level=30, error=repr(exc))
            return {"latency_metadata": {"llm.compute": round((time.perf_counter() - t0) * 1000, 1)}}
        # 100 is allowed so percentages can be expressed; every other number must be evidenced.
        evidence_numbers = [v for _, v in extract_numbers(evidence + " " + state["user_query"])] + [100.0]
        results = list(state["tool_results"])
        for calc in plan.calculations[:4]:
            # Reject expressions that use numbers not present in the evidence.
            if not all(number_supported(t, v, evidence_numbers) for t, v in extract_numbers(calc.expression)):
                log_event("compute_rejected_expression", level=30, expression=calc.expression)
                continue
            results.append(deps.tools["calculator"](expression=calc.expression, label=calc.label).model_dump())
        return {"tool_results": results,
                "latency_metadata": {"llm.compute": round((time.perf_counter() - t0) * 1000, 1)}}

    # ---------- check_evidence ----------
    @timed("check_evidence")
    def check_evidence(state: AgentState, config: RunnableConfig) -> dict:
        tid = _thread_id(config)
        plan_tools = {st["tool"] for st in state["plan"]}
        chunks = list(state.get("retrieved_documents", []))
        update: dict[str, Any] = {}
        notes: list[str] = []

        if "document_search" in plan_tools and not chunks:
            # Corrective retrieval: retry once with a keyword-only form of the question.
            keywords = " ".join(tokenize(state["user_query"]))
            if keywords:
                r = deps.retriever.retrieve(tid, keywords)
                chunks = r.chunks
                update["retrieval_attempts"] = state.get("retrieval_attempts", 0) + 1
                if chunks:
                    update["retrieved_documents"] = chunks
                    update["sources"] = _sources_from(chunks, state["tool_results"], state.get("media_evidence"))
                    log_event("corrective_retrieval_succeeded", chunks=len(chunks))

        media_evidence = list(state.get("media_evidence", []))
        if "video_search" in plan_tools and not media_evidence and deps.media is not None:
            keywords = " ".join(tokenize(state["user_query"]))
            if keywords:
                r = deps.media.search(tid, keywords,
                                      media_ids=state.get("classification", {}).get("media_ids") or None)
                if r.chunks:
                    media_evidence = r.chunks
                    update["retrieved_media"] = r.chunks
                    update["media_evidence"] = r.chunks
                    update["sources"] = _sources_from(update.get("retrieved_documents", chunks),
                                                      state["tool_results"], r.chunks)
                    log_event("corrective_media_retrieval_succeeded", chunks=len(r.chunks))
        insight_notes = [n for r in state["tool_results"] if r["tool"] == "media_insights"
                         for n in r["data"].get("empty_sections", [])]
        notes += insight_notes
        media_tools = plan_tools & _MEDIA_TOOLS
        if media_tools and not media_evidence and not insight_notes:
            if plan_tools <= {*_MEDIA_TOOLS, "calculator"}:
                return update | {
                    "final_response": prompts.INSUFFICIENT_MEDIA_EVIDENCE,
                    "verification_status": "INSUFFICIENT_EVIDENCE",
                    "evidence_score": 0.0,
                }
            notes.append("No relevant passage was found in the recordings for this question.")

        doc_only = plan_tools <= {"document_search", "calculator"} and "document_search" in plan_tools
        if "document_search" in plan_tools and not chunks:
            if doc_only:
                return update | {
                    "final_response": prompts.INSUFFICIENT_DOCUMENT_EVIDENCE,
                    "verification_status": "INSUFFICIENT_EVIDENCE",
                    "evidence_score": 0.0,
                }
            notes.append("No relevant passage was found in the uploaded document for this question.")

        data_results = [r for r in state["tool_results"] if r["tool"] not in _EVIDENCE_TOOLS]
        failed = [r for r in data_results if r["status"] != "success"]
        notes += [f"{r['tool']} failed: {r['error']}" for r in failed]
        sources = update.get("sources", state.get("sources", []))
        if insight_notes and not sources:  # "no decisions were identified" is itself the evidence
            return update | {"evidence_score": 1.0, "evidence_notes": notes}
        calc_ok = any(r["tool"] == "calculator" and r["status"] == "success" for r in state["tool_results"])
        if not sources and not calc_ok:
            msgs = list(dict.fromkeys(r["error"] for r in failed if r.get("error")))
            return update | {
                "final_response": (" ".join(msgs) or "I couldn't gather any evidence for this request.")
                + " I won't guess at live or document facts without evidence — please try again shortly.",
                "verification_status": "INSUFFICIENT_EVIDENCE",
                "evidence_score": 0.0,
            }

        components = []
        if "document_search" in plan_tools:
            components.append(max((c["dense_score"] for c in chunks), default=0.0))
        if media_tools:
            scored = [c["dense_score"] for c in media_evidence if c.get("dense_score") is not None]
            components.append(max(scored) if scored else (1.0 if media_evidence or insight_notes else 0.0))
        if data_results:
            components.append(1 - len(failed) / len(data_results))
        score = round(sum(components) / len(components), 3) if components else 1.0
        return update | {"evidence_score": score, "evidence_notes": notes}

    def route_after_evidence(state: AgentState) -> str:
        return "finalize_response" if state.get("final_response") else "generate_response"

    # ---------- generate_response ----------
    @timed("generate_response")
    def generate_response(state: AgentState, config: RunnableConfig) -> dict:
        attempt = state.get("attempts", 0) + 1
        plan_tools = [st["tool"] for st in state.get("plan", [])]
        calcs = [r for r in state.get("tool_results", []) if r["tool"] == "calculator"]

        # Pure calculation: answer deterministically — no LLM needed, no rounding risk.
        if plan_tools and set(plan_tools) == {"calculator"} and calcs:
            lines = [
                f"**{r['data']['expression']} = {_fmt_num(r['data']['result'])}**" if r["status"] == "success"
                else f"I couldn't compute that: {r['error']}"
                for r in calcs
            ]
            return {"draft_response": "\n\n".join(lines), "attempts": attempt}

        today = date.today().isoformat()
        history = conversation_history(state["messages"], s.history_turns, s.history_message_chars)
        if not state.get("plan"):
            messages = [SystemMessage(content=prompts.GENERAL_SYSTEM.format(today=today)), *history,
                        HumanMessage(content=state["user_query"])]
        else:
            calc_block = ""
            ok_calcs = [r for r in calcs if r["status"] == "success"]
            if ok_calcs:
                calc_block = "Calculation results (exact, from the calculator tool):\n" + "\n".join(
                    f"- {r['data'].get('label') or 'result'}: {r['data']['expression']} = {r['data']['result']}"
                    for r in ok_calcs) + "\n\n"
            notes = list(state.get("evidence_notes") or [])
            kinds = {src["type"] for src in state.get("sources", [])}
            if {"video", "document"} <= kinds:
                notes.append(prompts.CROSS_SOURCE_NOTE)
            notes_block = ("Notes:\n" + "\n".join(f"- {n}" for n in notes) + "\n\n") if notes else ""
            evidence = format_sources_for_prompt(state.get("sources", [])) or "(no sources)"
            user = prompts.ANSWER_USER.format(evidence=evidence, calculations=calc_block, notes=notes_block,
                                              query=state["user_query"])
            if attempt > 1 and state.get("verification_notes"):
                user += prompts.RETRY_SUFFIX.format(issues="; ".join(state["verification_notes"]))
            messages = [SystemMessage(content=prompts.ANSWER_SYSTEM.format(today=today)), *history,
                        HumanMessage(content=user)]

        t0 = time.perf_counter()
        response = _llm_call(lambda: deps.llm.invoke(messages, config=config))
        ms = round((time.perf_counter() - t0) * 1000, 1)
        metrics.observe("llm.generate", ms)
        text = _THINK.sub("", str(response.content)).strip()
        return {"draft_response": text, "attempts": attempt, "latency_metadata": {f"llm.generate#{attempt}": ms}}

    # ---------- verify_response ----------
    @timed("verify_response")
    def verify_response(state: AgentState, config: RunnableConfig) -> dict:
        draft = state.get("draft_response", "")
        sources = state.get("sources", [])
        rendered, citations, invalid = apply_citations(draft, sources)
        if not state.get("plan"):
            return {"final_response": rendered, "citations": [], "verification_status": "SKIPPED",
                    "verification_notes": []}

        # An explicit "the sources don't contain this" is a correct, honest outcome — it
        # needs no citation and must not be retried into a fabricated answer.
        abstained = is_abstention(draft) and not citations
        issues: list[str] = []
        if not draft.strip():
            issues.append("empty answer")
        if invalid:
            issues.append(f"citations to non-existent sources removed: {', '.join(invalid)}")
        if sources and not citations and not abstained:
            issues.append("no citations: cite the sources you used with their ids, e.g. [S1]")
        unsupported = check_numbers(draft, dict(state))
        if unsupported:
            issues.append(f"figures not found in the sources or calculations: {', '.join(unsupported[:6])}")
        unsupported_claims = check_claim_support(draft, sources)
        if unsupported_claims:
            issues.append("statements not supported by the sources they cite: "
                          + "; ".join(f'"{c}"' for c in unsupported_claims[:3]))

        if not issues:
            status = "INSUFFICIENT_EVIDENCE" if abstained else "VERIFIED"
        elif state.get("attempts", 1) < s.max_verify_attempts:
            status = "RETRY"
        else:
            status = "FAILED"
        log_event("verification", status=status, issues=issues, attempt=state.get("attempts"))
        return {"final_response": rendered, "citations": citations, "verification_status": status,
                "verification_notes": issues}

    def route_after_verify(state: AgentState) -> str:
        return "generate_response" if state["verification_status"] == "RETRY" else "finalize_response"

    # ---------- finalize_response ----------
    @timed("finalize_response")
    def finalize_response(state: AgentState, config: RunnableConfig) -> dict:
        text = state.get("final_response") or state.get("draft_response") or ""
        status = state.get("verification_status", "SKIPPED")
        if state.get("query_type") == "clarification_required" and not text:
            text = state.get("classification", {}).get("clarification_question") or "Could you clarify your request?"
            status = "SKIPPED"
        citations = list(state.get("citations", []))
        if status == "FAILED":
            doc_sources = [src for src in state.get("sources", []) if src["type"] in ("document", "video")]
            if doc_sources and not any(c["type"] in ("document", "video") for c in citations):
                text += "\n\n**Sources consulted:** " + "; ".join(
                    dict.fromkeys(citation_record(src)["display"] for src in doc_sources))
            text += ("\n\n> ⚠️ Verification note: some statements or figures could not be matched to "
                     "the retrieved sources. Treat them with caution.")
        state_view = {**state, "verification_status": status}
        lat = {k: v for k, v in (state.get("latency_metadata") or {}).items() if not k.startswith("_")}
        node_total = sum(v for k, v in lat.items() if "." not in k and "#" not in k)
        tools_used = list(dict.fromkeys(r["tool"] for r in state.get("tool_results", [])))
        finance = [
            {"tool": r["tool"], "data": r["data"], "source": r["source"]}
            for r in state.get("tool_results", [])
            if r["tool"] in ("stock_quote", "price_history") and r["status"] == "success"
        ]
        meta = {
            "route": state.get("query_type"),
            "plan": [{"tool": st["tool"], "input": st["input"], "period": st.get("period")} for st in state.get("plan", [])],
            "plan_summary": state.get("plan_summary", ""),
            "tools_used": tools_used,
            "citations": citations,
            "verification_status": status,
            "verification_notes": state.get("verification_notes", []),
            "evidence_score": state.get("evidence_score", 0.0),
            "retrieved_chunks": len(state.get("retrieved_documents", [])),
            "retrieved_media_chunks": len(state.get("retrieved_media", [])),
            "media_ids": state.get("classification", {}).get("media_ids", []),
            "latency_ms": round(node_total, 1),
            "latency_breakdown": lat,
            "timeline": build_timeline(state_view),
            "finance": finance,
            "attempts": state.get("attempts", 0),
            "classification_method": state.get("classification", {}).get("method"),
        }
        metrics.incr(f"route.{meta['route']}")
        metrics.incr(f"verification.{status}")
        metrics.observe("turn", float(node_total))
        log_event("turn_completed", route=meta["route"], tools=tools_used, verification=status,
                  retrieved_chunks=meta["retrieved_chunks"], latency_ms=meta["latency_ms"],
                  evidence_score=meta["evidence_score"], error=state.get("error_state"))
        return {
            "final_response": text,
            "verification_status": status,
            "messages": [AIMessage(content=text, additional_kwargs={"nova": meta})],
        }

    # ---------- wiring ----------
    g = StateGraph(AgentState)
    for name, fn in [
        ("validate_input", validate_input), ("classify_query", classify_query), ("planner", planner),
        ("approval_gate", approval_gate), ("execute_tools", execute_tools), ("compute", compute),
        ("check_evidence", check_evidence), ("generate_response", generate_response),
        ("verify_response", verify_response), ("finalize_response", finalize_response),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "validate_input")
    g.add_conditional_edges("validate_input", lambda st: "finalize_response" if st.get("error_state") else "classify_query",
                            ["finalize_response", "classify_query"])
    g.add_conditional_edges("classify_query", route_after_classify, ["generate_response", "finalize_response", "planner"])
    g.add_conditional_edges("planner", route_after_planner,
                            ["finalize_response", "generate_response", "approval_gate", "execute_tools"])
    g.add_conditional_edges("approval_gate", route_after_approval, ["finalize_response", "execute_tools"])
    g.add_edge("execute_tools", "compute")
    g.add_edge("compute", "check_evidence")
    g.add_conditional_edges("check_evidence", route_after_evidence, ["finalize_response", "generate_response"])
    g.add_edge("generate_response", "verify_response")
    g.add_conditional_edges("verify_response", route_after_verify, ["generate_response", "finalize_response"])
    g.add_edge("finalize_response", END)
    return g


def summarize_sources_text(sources: list[dict[str, Any]]) -> str:
    """Evidence text for evaluation judges."""
    return json.dumps([{k: v for k, v in s.items() if k in ("id", "type", "filename", "page", "url", "label", "text")}
                       for s in sources], ensure_ascii=False)
