"""Service-level integration: persistence, isolation, deletion, error translation."""

import httpx
import pytest

from nova.agent.graph import AgentDeps
from nova.errors import LLMUnavailableError, ThreadNotFoundError
from nova.memory import ThreadRegistry, connect, make_checkpointer
from nova.service import NovaService
from tests.agent.test_graph import DOC_ROUTE, grounded_answer, router


def test_conversation_survives_restart(make_service):
    svc = make_service(structured_fn=router({}), answer_fn=lambda m: "Hello!")
    svc.chat("hi there, remember the number 42", "persist-1")
    # Simulate a process restart: new connections, new graph, same database file.
    s = svc.settings
    restarted = NovaService(AgentDeps(llm=svc.deps.llm, store=svc.store, retriever=svc.deps.retriever, settings=s),
                            make_checkpointer(connect(s.db_path)), ThreadRegistry(connect(s.db_path)))
    thread = restarted.get_thread("persist-1")
    assert [m["role"] for m in thread["messages"]] == ["user", "assistant"]
    assert thread["messages"][0]["content"] == "hi there, remember the number 42"
    assert restarted.list_threads()[0]["thread_id"] == "persist-1"


def test_threads_are_isolated(make_service):
    seen = []
    svc = make_service(structured_fn=router({}), answer_fn=lambda m: seen.append([str(x.content) for x in m]) or "ok")
    svc.chat("secret code is ALPHA", "thread-a")
    svc.chat("what is the secret code?", "thread-b")
    assert not any("ALPHA" in c for c in seen[-1])


def test_document_isolation_between_threads(make_service, report_bytes):
    svc = make_service(structured_fn=router(DOC_ROUTE),
                       answer_fn=grounded_answer({"28.1": "Net income was $28.1 million [S1]."}))
    svc.ingest_document("thread-a", "report.pdf", report_bytes)
    a = svc.chat("What was net income per the report?", "thread-a")
    b = svc.chat("What was net income per the report?", "thread-b")
    assert "28.1" in a.answer
    assert "28.1" not in b.answer and svc.list_documents("thread-b") == []


def test_duplicate_upload_is_deduplicated(make_service, report_bytes):
    svc = make_service()
    first = svc.ingest_document("t", "report.pdf", report_bytes)
    second = svc.ingest_document("t", "report-copy.pdf", report_bytes)
    assert not first["duplicate"] and second["duplicate"] and second["document_id"] == first["document_id"]


def test_delete_thread_removes_everything(make_service, report_bytes):
    svc = make_service(structured_fn=router({}), answer_fn=lambda m: "ok")
    svc.ingest_document("gone", "report.pdf", report_bytes)
    svc.chat("hello", "gone")
    svc.delete_thread("gone")
    assert svc.registry.get("gone") is None
    assert svc.list_documents("gone") == []
    assert svc._messages("gone") == []
    with pytest.raises(ThreadNotFoundError):
        svc.get_thread("gone")
    with pytest.raises(ThreadNotFoundError):
        svc.delete_thread("gone")


def test_thread_title_from_first_message_even_after_upload(make_service, report_bytes):
    svc = make_service(structured_fn=router({}), answer_fn=lambda m: "ok")
    svc.ingest_document("t", "report.pdf", report_bytes)
    assert svc.registry.get("t")["title"] == "📄 report.pdf"
    svc.chat("Summarise the outlook", "t")
    svc.chat("Thanks", "t")
    assert svc.registry.get("t")["title"] == "Summarise the outlook"


def test_llm_down_raises_friendly_error(make_service):
    def down(*a, **k):
        raise httpx.ConnectError("[WinError 10061] No connection could be made")

    svc = make_service(structured_fn=down, answer_fn=down)
    with pytest.raises(LLMUnavailableError) as exc:
        svc.chat("Tell me something general about markets please", "t")
    assert "Ollama" in exc.value.user_message


def test_invalid_thread_id_rejected(make_service):
    from nova.errors import InputValidationError

    svc = make_service()
    with pytest.raises(InputValidationError):
        svc.chat("hi", "../../etc/passwd")
    with pytest.raises(InputValidationError):
        svc.ingest_document("bad id!", "r.pdf", b"%PDF-")


def test_readiness_reports_llm_down(make_service, monkeypatch):
    svc = make_service()
    monkeypatch.setattr("nova.service.check_llm", lambda s: (False, "Nova cannot reach the local model server"))
    report = svc.readiness()
    assert report["ready"] is False and report["checks"]["database"]["ok"] is True


def test_registry_backfill_after_migration_does_not_deadlock(make_service):
    """Regression: backfilling the thread registry from existing checkpoints used to deadlock
    (iterating SqliteSaver.list() while reading state takes the same non-reentrant lock)."""
    import concurrent.futures

    svc = make_service(structured_fn=router({}), answer_fn=lambda m: "ok")
    svc.chat("first conversation", "legacy-1")
    svc.chat("second conversation", "legacy-2")
    with svc.registry._lock, svc.registry._conn:  # simulate a pre-registry (migrated) database
        svc.registry._conn.execute("DELETE FROM threads")
    s = svc.settings
    fresh = NovaService(AgentDeps(llm=svc.deps.llm, store=svc.store, retriever=svc.deps.retriever, settings=s),
                        make_checkpointer(connect(s.db_path)), ThreadRegistry(connect(s.db_path)))
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        pool.submit(fresh._backfill_registry).result(timeout=20)
    assert {t["thread_id"] for t in fresh.list_threads()} == {"legacy-1", "legacy-2"}
    assert {t["title"] for t in fresh.list_threads()} == {"first conversation", "second conversation"}
