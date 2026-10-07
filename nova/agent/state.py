"""
Typed LangGraph state.

Only `messages` accumulates across turns (via `add_messages`). Every other field is
per-turn working memory that `validate_input` resets at the start of each turn, so
nothing stale leaks from one question into the next. All values are plain JSON-able
data (dicts/lists/str/float) — never live objects — so checkpoints stay portable.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

QueryType = Literal[
    "general", "web", "finance", "document", "video", "calculation", "multi_tool", "clarification_required"
]
VerificationStatus = Literal["PENDING", "VERIFIED", "INSUFFICIENT_EVIDENCE", "RETRY", "FAILED", "SKIPPED"]


def merge_latency(old: dict[str, float] | None, new: dict[str, float] | None) -> dict[str, float]:
    """Reducer: merge timing dicts; a `_reset` key starts a fresh dict for a new turn."""
    new = new or {}
    if new.get("_reset"):
        return {k: v for k, v in new.items() if k != "_reset"}
    return {**(old or {}), **new}


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    # ---- per-turn ----
    user_query: str
    query_type: QueryType
    classification: dict[str, Any]
    plan: list[dict[str, Any]]
    plan_summary: str
    needs_calculation: bool
    selected_tools: list[str]
    requires_human_approval: bool
    approval: dict[str, Any] | None
    tool_results: list[dict[str, Any]]
    retrieved_documents: list[dict[str, Any]]
    retrieved_media: list[dict[str, Any]]  # ranked transcript chunks (temporal retrieval)
    media_evidence: list[dict[str, Any]]  # insight items + transcript chunks given to the model
    active_media: list[str]  # NOT reset per turn: recordings the conversation is about (memory)
    sources: list[dict[str, Any]]
    retrieval_attempts: int
    evidence_score: float
    evidence_notes: list[str]
    draft_response: str
    attempts: int
    verification_status: VerificationStatus
    verification_notes: list[str]
    citations: list[dict[str, Any]]
    latency_metadata: Annotated[dict[str, float], merge_latency]
    error_state: dict[str, Any] | None
    final_response: str


PER_TURN_DEFAULTS: dict[str, Any] = {
    "query_type": "general",
    "classification": {},
    "plan": [],
    "plan_summary": "",
    "needs_calculation": False,
    "selected_tools": [],
    "requires_human_approval": False,
    "approval": None,
    "tool_results": [],
    "retrieved_documents": [],
    "retrieved_media": [],
    "media_evidence": [],
    "sources": [],
    "retrieval_attempts": 0,
    "evidence_score": 0.0,
    "evidence_notes": [],
    "draft_response": "",
    "attempts": 0,
    "verification_status": "PENDING",
    "verification_notes": [],
    "citations": [],
    "error_state": None,
    "final_response": "",
}
