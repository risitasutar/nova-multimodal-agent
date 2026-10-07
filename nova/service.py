"""
NovaService — the single façade over the agent used by BOTH the Streamlit UI and the
FastAPI app. Agent logic lives in the graph; this layer adds request context,
validation, concurrency control per thread, error translation and thread/document
management.
"""

from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from typing import Any, cast

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Command

from nova.agent.graph import AgentDeps, build_graph
from nova.config import PROJECT_ROOT, Settings, get_settings
from nova.errors import (
    ApprovalStateError,
    InputValidationError,
    LLMUnavailableError,
    NovaError,
    ThreadNotFoundError,
    is_connection_error,
)
from nova.llm import check_llm, get_chat_model, get_embeddings
from nova.memory import ThreadRegistry, connect, make_checkpointer
from nova.observability import configure_logging, configure_tracing, log_event, metrics, request_context
from nova.rag.ingest import parse_pdf
from nova.rag.retrieve import Retriever
from nova.rag.store import DocumentStore
from nova.safety import clean_user_message

STEP_LABELS = {
    "validate_input": "Input validated",
    "classify_query": "Query analyzed",
    "planner": "Plan created",
    "approval_gate": "Approval recorded",
    "execute_tools": "Tools executed",
    "compute": "Calculations checked",
    "check_evidence": "Evidence checked",
    "generate_response": "Response drafted",
    "verify_response": "Answer verified",
    "finalize_response": "Done",
}


@dataclass
class TurnResult:
    thread_id: str
    request_id: str
    status: str  # completed | awaiting_approval
    answer: str = ""
    route: str | None = None
    plan: list[dict[str, Any]] = field(default_factory=list)
    plan_summary: str = ""
    tools_used: list[str] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    verification_status: str | None = None
    evidence_score: float = 0.0
    latency_ms: float = 0.0
    wall_time_ms: float = 0.0
    timeline: list[str] = field(default_factory=list)
    finance: list[dict[str, Any]] = field(default_factory=list)
    approval: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_valid_thread_id(thread_id: str) -> bool:
    return bool(thread_id) and len(thread_id) <= 64 and all(c.isalnum() or c in "-_" for c in thread_id)


class NovaService:
    def __init__(
        self,
        deps: AgentDeps,
        checkpointer: BaseCheckpointSaver,
        registry: ThreadRegistry,
    ) -> None:
        self.deps = deps
        self.settings = deps.settings
        self.store = deps.store
        self.media = deps.media  # MediaService (video/audio) — shared with the agent's tools
        self.registry = registry
        self.checkpointer = checkpointer
        self.graph = build_graph(deps).compile(checkpointer=checkpointer)
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    # ------------------------------------------------------------ construction
    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> NovaService:
        s = settings or get_settings()
        configure_logging(s.log_level, s.log_json, s.log_dir)
        configure_tracing(s)
        _migrate_legacy_db(s)
        conn = connect(s.db_path)
        checkpointer = make_checkpointer(conn)
        registry = ThreadRegistry(connect(s.db_path))
        embeddings = get_embeddings(s)
        store = DocumentStore(s.vector_dir, embeddings, s.embedding_model_name)
        retriever = Retriever(
            store, embeddings, top_k=s.retrieval_top_k, candidates=s.retrieval_candidates,
            min_relevance=s.min_relevance,
        )
        llm = get_chat_model(s)
        media = build_media_service(s, embeddings, llm)
        service = cls(AgentDeps(llm=llm, store=store, retriever=retriever, settings=s, media=media),
                      checkpointer, registry)
        service._backfill_registry()
        return service

    def _backfill_registry(self) -> None:
        """One-time: register threads that exist only as checkpoints (pre-registry data)."""
        if not self.registry.is_empty():
            return
        # Materialise the ids first: `checkpointer.list()` is a generator that holds SqliteSaver's
        # (non-reentrant) lock while iterating, so reading state inside the loop would deadlock.
        seen = list(dict.fromkeys(str(cp.config["configurable"]["thread_id"]) for cp in self.checkpointer.list(None)))
        for tid in seen:
            msgs = self._messages(tid)
            first = next((m.content for m in msgs if isinstance(m, HumanMessage)), None)
            if first:
                self.registry.touch(tid, str(first))
        if seen:
            log_event("registry_backfilled", threads=len(seen))

    # ------------------------------------------------------------ helpers
    def _lock_for(self, thread_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(thread_id, threading.Lock())

    @staticmethod
    def _config(thread_id: str, approval_required: bool | None, request_id: str) -> RunnableConfig:
        return {
            "configurable": {"thread_id": thread_id, "approval_required": approval_required},
            "metadata": {"thread_id": thread_id, "request_id": request_id},
            "run_name": "nova_turn",
            "recursion_limit": 40,
        }

    def _messages(self, thread_id: str) -> list[Any]:
        state = self.graph.get_state({"configurable": {"thread_id": thread_id}})
        return list(state.values.get("messages", [])) if state and state.values else []

    def pending_approval(self, thread_id: str) -> dict[str, Any] | None:
        state = self.graph.get_state({"configurable": {"thread_id": thread_id}})
        for task in getattr(state, "tasks", ()) or ():
            for intr in getattr(task, "interrupts", ()) or ():
                return dict(intr.value)
        return None

    # ------------------------------------------------------------ chat
    def stream(
        self,
        message: str | None,
        thread_id: str | None = None,
        *,
        approval_required: bool | None = None,
        resume: dict[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """
        Run one turn (or resume an approval) and yield UI events:
          {"type": "step", "node", "label"} | {"type": "token", "text"} |
          {"type": "retry"} | {"type": "result", "result": TurnResult}
        """
        thread_id = thread_id or str(uuid.uuid4())
        if not _is_valid_thread_id(thread_id):
            raise InputValidationError(f"bad thread id {thread_id!r}", user_message="Invalid conversation id.")
        with request_context(thread_id=thread_id) as request_id, self._lock_for(thread_id):
            start = time.perf_counter()
            pending = self.pending_approval(thread_id)
            if resume is not None:
                if not pending:
                    raise ApprovalStateError(f"no pending approval on {thread_id}")
                graph_input: Any = Command(resume=resume)
            else:
                if pending:
                    raise ApprovalStateError(
                        "approval pending",
                        user_message="This conversation is waiting for your approval. Approve or reject it first.",
                    )
                text = clean_user_message(message)
                graph_input = {"messages": [HumanMessage(content=text)]}
                self.registry.touch(thread_id, text)
            config = self._config(thread_id, approval_required, request_id)
            metrics.incr("requests.total")
            log_event("turn_started", resume=resume is not None)
            try:
                for item in self.graph.stream(graph_input, config=config, stream_mode=["updates", "messages"]):
                    mode, payload = cast(tuple[str, Any], item)
                    if mode == "messages":
                        chunk, meta = payload
                        if meta.get("langgraph_node") == "generate_response" and getattr(chunk, "content", ""):
                            yield {"type": "token", "text": chunk.content}
                        continue
                    for node, update in payload.items():
                        if node == "__interrupt__":
                            continue
                        yield {"type": "step", "node": node, "label": STEP_LABELS.get(node, node)}
                        if node == "verify_response" and (update or {}).get("verification_status") == "RETRY":
                            yield {"type": "retry"}
            except NovaError:
                metrics.incr("requests.error")
                raise
            except Exception as exc:
                metrics.incr("requests.error")
                if is_connection_error(exc):
                    log_event("llm_unreachable", level=40, error=repr(exc))
                    raise LLMUnavailableError(repr(exc)) from exc
                log_event("turn_failed", level=40, error=repr(exc), exc_info=True)
                raise NovaError(repr(exc)) from exc
            result = self._result(thread_id, request_id, (time.perf_counter() - start) * 1000)
            yield {"type": "result", "result": result}

    def _result(self, thread_id: str, request_id: str, wall_ms: float) -> TurnResult:
        approval = self.pending_approval(thread_id)
        if approval:
            state = self.graph.get_state({"configurable": {"thread_id": thread_id}}).values
            return TurnResult(thread_id=thread_id, request_id=request_id, status="awaiting_approval",
                              approval=approval, route=state.get("query_type"), plan=state.get("plan", []),
                              plan_summary=state.get("plan_summary", ""), wall_time_ms=round(wall_ms, 1))
        last = next((m for m in reversed(self._messages(thread_id)) if isinstance(m, AIMessage)), None)
        meta = (last.additional_kwargs.get("nova") if last else None) or {}
        return TurnResult(
            thread_id=thread_id, request_id=request_id, status="completed",
            answer=str(last.content) if last else "",
            route=meta.get("route"), plan=meta.get("plan", []), plan_summary=meta.get("plan_summary", ""),
            tools_used=meta.get("tools_used", []), citations=meta.get("citations", []),
            verification_status=meta.get("verification_status"), evidence_score=meta.get("evidence_score", 0.0),
            latency_ms=meta.get("latency_ms", 0.0), wall_time_ms=round(wall_ms, 1),
            timeline=meta.get("timeline", []), finance=meta.get("finance", []),
        )

    def chat(self, message: str, thread_id: str | None = None, *, approval_required: bool | None = None) -> TurnResult:
        result: TurnResult | None = None
        for event in self.stream(message, thread_id, approval_required=approval_required):
            if event["type"] == "result":
                result = event["result"]
        assert result is not None
        return result

    def resume(self, thread_id: str, approved: bool) -> TurnResult:
        result: TurnResult | None = None
        for event in self.stream(None, thread_id, resume={"approved": approved}):
            if event["type"] == "result":
                result = event["result"]
        assert result is not None
        return result

    # ------------------------------------------------------------ documents
    def ingest_document(self, thread_id: str, filename: str | None, data: bytes) -> dict[str, Any]:
        if not _is_valid_thread_id(thread_id):
            raise InputValidationError(f"bad thread id {thread_id!r}", user_message="Invalid conversation id.")
        with request_context(thread_id=thread_id):
            existing = self.store.list_documents(thread_id)
            if len(existing) >= self.settings.max_documents_per_thread:
                raise InputValidationError(
                    "document limit", user_message=f"A conversation can hold at most {self.settings.max_documents_per_thread} documents."
                )
            start = time.perf_counter()
            parsed = parse_pdf(data, filename)
            duplicate = self.store.find_by_hash(thread_id, parsed.file_sha256)
            if duplicate:
                return duplicate | {"duplicate": True}
            try:
                info = self.store.add_document(thread_id, parsed)
            except Exception as exc:
                if is_connection_error(exc):
                    raise LLMUnavailableError(repr(exc)) from exc
                raise
            ms = round((time.perf_counter() - start) * 1000, 1)
            metrics.observe("ingest", ms)
            flagged = sum(1 for c in parsed.chunks if c.injection_flags)
            log_event("document_ingested", filename=parsed.filename, pages=parsed.pages,
                      chunks=len(parsed.chunks), injection_flagged_chunks=flagged, latency_ms=ms)
            self.registry.touch(thread_id, f"📄 {parsed.filename}", count_turn=False)
            return info | {"duplicate": False, "ingest_ms": ms, "injection_flagged_chunks": flagged}

    def list_documents(self, thread_id: str) -> list[dict[str, Any]]:
        return self.store.list_documents(thread_id) if _is_valid_thread_id(thread_id) else []

    # ------------------------------------------------------------ threads
    def list_threads(self, limit: int = 100) -> list[dict[str, Any]]:
        return self.registry.list(limit)

    def get_thread(self, thread_id: str) -> dict[str, Any]:
        meta = self.registry.get(thread_id) if _is_valid_thread_id(thread_id) else None
        if meta is None:
            raise ThreadNotFoundError(thread_id)
        messages = []
        for m in self._messages(thread_id):
            if isinstance(m, HumanMessage):
                messages.append({"role": "user", "content": str(m.content)})
            elif isinstance(m, AIMessage) and m.content and not m.tool_calls:
                messages.append({"role": "assistant", "content": str(m.content),
                                 "metadata": m.additional_kwargs.get("nova", {})})
        return meta | {
            "messages": messages,
            "documents": self.list_documents(thread_id),
            "media": [a.model_dump() for a in self.media.list_assets(thread_id)] if self.media else [],
            "pending_approval": self.pending_approval(thread_id),
        }

    def delete_thread(self, thread_id: str) -> None:
        if not _is_valid_thread_id(thread_id) or self.registry.get(thread_id) is None:
            raise ThreadNotFoundError(thread_id)
        with self._lock_for(thread_id):
            self.checkpointer.delete_thread(thread_id)
            self.store.delete_thread(thread_id)
            if self.media is not None:
                self.media.delete_thread(thread_id)
            self.registry.delete(thread_id)
        log_event("thread_deleted", thread_id=thread_id)

    # ------------------------------------------------------------ health
    def readiness(self) -> dict[str, Any]:
        llm_ok, llm_msg = check_llm(self.settings)
        try:
            with sqlite3.connect(str(self.settings.db_path), timeout=2) as conn:
                conn.execute("SELECT 1")
            db_ok, db_msg = True, "ok"
        except sqlite3.Error as exc:
            db_ok, db_msg = False, repr(exc)
        return {
            "ready": llm_ok and db_ok,
            "checks": {"llm": {"ok": llm_ok, "detail": llm_msg}, "database": {"ok": db_ok, "detail": db_msg}},
            "model": self.settings.chat_model_name,
            "embedding_model": self.settings.embedding_model_name,
        }


def build_media_service(s: Settings, embeddings: Any, llm: Any) -> Any:
    """Media capability over the SAME embeddings, FAISS store implementation and database."""
    from nova.video.service import MediaService
    from nova.video.storage import MediaRegistry

    media_store = DocumentStore(s.media_vector_dir, embeddings, s.embedding_model_name)
    media_retriever = Retriever(media_store, embeddings, top_k=s.retrieval_top_k, candidates=s.retrieval_candidates,
                                min_relevance=s.video_min_relevance)
    return MediaService(s, MediaRegistry(connect(s.db_path)), media_store, media_retriever, llm)


def _migrate_legacy_db(s: Settings) -> None:
    """Carry conversations over from the pre-upgrade `chatbot.db` (one-time, non-destructive)."""
    legacy = PROJECT_ROOT / "chatbot.db"
    if s.db_path.exists() or not legacy.exists() or s.environment == "test":
        return
    s.db_path.parent.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(str(legacy))
    dst = sqlite3.connect(str(s.db_path))
    try:
        src.backup(dst)
        log_event("legacy_db_migrated", source=str(legacy.name))
    finally:
        src.close()
        dst.close()


_service: NovaService | None = None
_service_lock = threading.Lock()


def get_service() -> NovaService:
    """Process-wide singleton (lazy: importing this module never touches Ollama)."""
    global _service
    with _service_lock:
        if _service is None:
            _service = NovaService.from_settings()
        return _service
