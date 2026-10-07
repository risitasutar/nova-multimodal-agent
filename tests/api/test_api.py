"""HTTP API tests: FastAPI TestClient against the real service wired to fakes."""

import pytest
from fastapi.testclient import TestClient

from nova.config import get_settings
from tests.agent.test_graph import DOC_ROUTE, grounded_answer, router


@pytest.fixture
def client(make_service, monkeypatch):
    from api import main

    svc = make_service(structured_fn=router(DOC_ROUTE),
                       answer_fn=grounded_answer({"28.1": "Net income was $28.1 million [S1].",
                                                  "333.69": "AAPL is at 333.69 USD [F1]."}))
    main.app.dependency_overrides[main.service_dep] = lambda: svc
    monkeypatch.setattr("nova.service.check_llm", lambda s: (True, "test-model ready"))
    with TestClient(main.app, raise_server_exceptions=False) as c:
        yield c
    main.app.dependency_overrides.clear()


def test_health_and_ready(client):
    assert client.get("/health").json()["status"] == "ok"
    ready = client.get("/ready")
    assert ready.status_code == 200 and ready.json()["ready"] is True
    assert "x-request-id" in ready.headers


def test_chat_new_thread_and_metrics(client):
    r = client.post("/chat", json={"message": "What is 9283 * 47?"})
    body = r.json()
    assert r.status_code == 200 and body["status"] == "completed"
    assert body["answer"] == "**9283 * 47 = 436,301**" and body["tools_used"] == ["calculator"]
    assert body["thread_id"] and body["latency_ms"] >= 0
    metrics = client.get("/metrics").json()
    assert metrics["counters"]["requests.total"] >= 1


def test_document_upload_then_grounded_chat(client, report_bytes):
    up = client.post("/documents", data={"thread_id": "api-t1"},
                     files={"file": ("report.pdf", report_bytes, "application/pdf")})
    assert up.status_code == 200 and up.json()["pages"] == 10
    r = client.post("/chat", json={"thread_id": "api-t1", "message": "What was net income per the report?"}).json()
    assert r["verification_status"] == "VERIFIED"
    assert r["citations"][0]["type"] == "document" and r["citations"][0]["page"] >= 1


def test_upload_validation_errors(client):
    bad = client.post("/documents", data={"thread_id": "t"}, files={"file": ("notes.txt", b"hi", "text/plain")})
    assert bad.status_code == 415 and bad.json()["error"]["code"] == "unsupported_file"
    broken = client.post("/documents", data={"thread_id": "t"},
                         files={"file": ("x.pdf", b"%PDF-1.4 garbage", "application/pdf")})
    assert broken.status_code == 422 and "Traceback" not in broken.text


def test_thread_endpoints(client):
    client.post("/chat", json={"thread_id": "api-t2", "message": "What is 2 + 2?"})
    threads = client.get("/threads").json()
    assert any(t["thread_id"] == "api-t2" for t in threads)
    detail = client.get("/threads/api-t2").json()
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]
    assert client.delete("/threads/api-t2").status_code == 204
    missing = client.get("/threads/api-t2")
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "thread_not_found"


def test_approval_flow_over_http(client):
    r = client.post("/chat", json={"thread_id": "api-t3", "message": "What is the current stock price of AAPL?",
                                   "approval_required": True}).json()
    assert r["status"] == "awaiting_approval" and r["approval"]["operations"]
    blocked = client.post("/chat", json={"thread_id": "api-t3", "message": "hello"})
    assert blocked.status_code == 409
    done = client.post("/chat/api-t3/approval", json={"approved": True}).json()
    assert done["status"] == "completed" and "333.69" in done["answer"]
    assert client.post("/chat/api-t3/approval", json={"approved": True}).status_code == 409


def test_input_validation(client):
    assert client.post("/chat", json={"message": ""}).status_code == 422
    too_long = client.post("/chat", json={"message": "x" * 5000})
    assert too_long.status_code == 422 and "too long" in too_long.json()["error"]["message"]
    assert client.post("/chat", json={"message": "hi", "thread_id": "../etc"}).status_code == 422


def test_api_key_enforced_when_configured(client, monkeypatch):
    monkeypatch.setenv("NOVA_API_KEY", "s3cret")
    get_settings.cache_clear()
    assert client.get("/threads").status_code == 401
    assert client.get("/threads", headers={"x-api-key": "wrong"}).status_code == 401
    assert client.get("/threads", headers={"x-api-key": "s3cret"}).status_code == 200
    assert client.get("/health").status_code == 200  # liveness stays open


def test_llm_unavailable_maps_to_503(client, monkeypatch):
    from nova.errors import LLMUnavailableError
    from nova.service import NovaService

    def boom(self, *a, **k):
        raise LLMUnavailableError("connect refused")

    monkeypatch.setattr(NovaService, "chat", boom)
    r = client.post("/chat", json={"message": "hello"})
    assert r.status_code == 503 and "Ollama" in r.json()["error"]["message"]
