"""Full LangGraph agent with scripted LLM + fake tools: routing, grounding, verification, HITL."""

import pytest

from nova.agent import prompts
from nova.agent.classifier import QueryClassification
from nova.agent.planner import Plan


def router(mapping):
    """Structured-output script: route by substring of the latest message."""

    def _fn(schema, msgs):
        text = str(msgs[-1].content)
        if schema is QueryClassification:
            for needle, out in mapping.items():
                if needle.lower() in text.lower():
                    return out
            return {"query_type": "general"}
        if schema is Plan:
            return mapping.get("__plan__", {"steps": []})
        return mapping.get(schema.__name__, {"calculations": []})

    return _fn


def grounded_answer(text_by_source):
    """Answer script: return a canned answer that cites the first source containing a needle."""

    def _fn(msgs):
        prompt = str(msgs[-1].content)
        for needle, answer in text_by_source.items():
            if needle in prompt:
                return answer
        return "I don't know."

    return _fn


DOC_ROUTE = {"net income": {"query_type": "document", "document_query": "net income 2025"}}


def test_document_question_is_grounded_and_cited(make_service, report_bytes):
    svc = make_service(structured_fn=router(DOC_ROUTE),
                       answer_fn=grounded_answer({"28.1": "Net income was $28.1 million in 2025 [S1]."}))
    svc.ingest_document("t1", "report.pdf", report_bytes)
    r = svc.chat("What was Northwind's net income in 2025?", "t1")
    assert r.route == "document" and r.tools_used == ["document_search"]
    assert r.verification_status == "VERIFIED"
    assert "[PDF: report.pdf, p." in r.answer and "[S1]" not in r.answer
    assert r.citations and r.citations[0]["filename"] == "report.pdf"
    assert any("document chunk" in step for step in r.timeline)


def test_no_evidence_means_no_confident_answer(make_service, report_bytes):
    svc = make_service(structured_fn=router({"recipe": {"query_type": "document", "document_query": "banana bread"}}),
                       answer_fn=lambda m: pytest.fail("LLM must not answer without evidence"),
                       min_relevance=0.9)
    svc.ingest_document("t1", "report.pdf", report_bytes)
    r = svc.chat("What banana bread recipe does the report recommend?", "t1")
    assert r.answer == prompts.INSUFFICIENT_DOCUMENT_EVIDENCE
    assert r.verification_status == "INSUFFICIENT_EVIDENCE" and r.citations == []


def test_document_question_without_upload(make_service):
    svc = make_service(structured_fn=router(DOC_ROUTE))
    r = svc.chat("What was net income according to my report?", "t1")
    assert r.answer == prompts.NO_DOCUMENT_UPLOADED


def test_verifier_retries_once_then_accepts_corrected_answer(make_service, report_bytes):
    answers = iter(["Net income was $31.4 million [S1].", "Net income was $28.1 million [S1]."])
    svc = make_service(structured_fn=router(DOC_ROUTE), answer_fn=lambda m: next(answers))
    svc.ingest_document("t1", "report.pdf", report_bytes)
    events = list(svc.stream("What was net income in 2025 per the report?", "t1"))
    assert any(e["type"] == "retry" for e in events)
    result = events[-1]["result"]
    assert result.verification_status == "VERIFIED" and "28.1" in result.answer and "31.4" not in result.answer
    meta = svc.get_thread("t1")["messages"][-1]["metadata"]
    assert meta["attempts"] == 2


def test_verifier_is_bounded_and_flags_unverified_output(make_service, report_bytes):
    calls = []

    def always_wrong(msgs):
        calls.append(1)
        return "Net income was $99.9 million [S7]."

    svc = make_service(structured_fn=router(DOC_ROUTE), answer_fn=always_wrong)
    svc.ingest_document("t1", "report.pdf", report_bytes)
    r = svc.chat("What was net income per the report?", "t1")
    assert len(calls) == 2  # NOVA_MAX_VERIFY_ATTEMPTS default
    assert r.verification_status == "FAILED"
    assert "Verification note" in r.answer and "Sources consulted" in r.answer and "[S7]" not in r.answer


def test_claim_citing_wrong_content_triggers_retry(make_service, report_bytes):
    answers = iter(["Zebras migrated across volcanic glaciers during the monsoon season [S1].",
                    "Net income was $28.1 million in 2025 [S1]."])
    svc = make_service(structured_fn=router(DOC_ROUTE), answer_fn=lambda m: next(answers))
    svc.ingest_document("t1", "report.pdf", report_bytes)
    events = list(svc.stream("What was net income in 2025 per the report?", "t1"))
    assert any(e["type"] == "retry" for e in events)
    result = events[-1]["result"]
    assert result.verification_status == "VERIFIED" and "Zebras" not in result.answer
    assert svc.get_thread("t1")["messages"][-1]["metadata"]["attempts"] == 2


def test_abstention_is_accepted_without_retry(make_service, report_bytes):
    calls = []

    def abstain(msgs):
        calls.append(1)
        return "The sources do not contain the dividend per share."

    svc = make_service(structured_fn=router({"dividend": {"query_type": "document", "document_query": "dividend"}}),
                       answer_fn=abstain, min_relevance=0.0)
    svc.ingest_document("t1", "report.pdf", report_bytes)
    r = svc.chat("What dividend did Northwind pay?", "t1")
    assert len(calls) == 1 and r.verification_status == "INSUFFICIENT_EVIDENCE"


def test_pure_calculation_needs_no_llm(make_service):
    svc = make_service(structured_fn=lambda *a: pytest.fail("no LLM"), answer_fn=lambda m: pytest.fail("no LLM"))
    r = svc.chat("What is 9283 * 47?", "t1")
    assert r.answer == "**9283 * 47 = 436,301**" and r.tools_used == ["calculator"]


def test_finance_route_uses_live_tool_and_cites_it(make_service):
    svc = make_service(answer_fn=grounded_answer({"333.69": "AAPL last traded at 333.69 USD [F1]."}))
    r = svc.chat("What is the current stock price of AAPL?", "t1")
    assert r.route == "finance" and r.tools_used == ["stock_quote"]
    assert "[FINANCE: AAPL, TestFeed, retrieved " in r.answer
    assert r.finance and r.finance[0]["data"]["price"] == 333.69
    assert r.verification_status == "VERIFIED"
    loc = r.citations[0]["locator"]
    assert loc["symbol"] == "AAPL" and loc["metric"] == "quote" and loc["retrieved_at"]


def test_failed_external_tool_degrades_gracefully(make_service):
    from nova.tools.base import ToolResult

    def down(symbol):
        return ToolResult(tool="stock_quote", status="error", error="Market data is temporarily unavailable.")

    svc = make_service(tools={"stock_quote": down}, answer_fn=lambda m: pytest.fail("must not guess"))
    r = svc.chat("What is the current stock price of AAPL?", "t1")
    assert "Market data is temporarily unavailable." in r.answer
    assert r.verification_status == "INSUFFICIENT_EVIDENCE"


def test_multi_tool_plan_with_computed_calculation(make_service, report_bytes):
    structured = router({
        "compare": {"query_type": "multi_tool", "tickers": ["AAPL"], "document_query": "net income 2025"},
        "__plan__": {"steps": [{"tool": "document_search", "input": "net income 2025"},
                               {"tool": "stock_quote", "input": "AAPL"}], "needs_calculation": True},
        "_CalculationPlan": {"calculations": [{"label": "ratio", "expression": "333.69 / 28.1"},
                                              {"label": "invented", "expression": "777 * 2"}]},
    })
    svc = make_service(structured_fn=structured,
                       answer_fn=grounded_answer({"<source": "Net income $28.1 million [S1]; AAPL 333.69 [F1]; ratio 11.875 ."}))
    svc.ingest_document("t1", "report.pdf", report_bytes)
    r = svc.chat("Compare AAPL's price with net income in my report", "t1")
    assert r.route == "multi_tool"
    assert {"document_search", "stock_quote", "calculator"} <= set(r.tools_used)
    calcs = [s for s in r.timeline if "Calculation completed" in s]
    assert len(calcs) == 1 and "333.69 / 28.1" in calcs[0]  # invented-number expression rejected
    assert r.verification_status == "VERIFIED"


def test_prompt_injection_in_document_is_quoted_as_data(make_service, injection_bytes):
    seen = {}

    def answer(msgs):
        seen["system"] = str(msgs[0].content)
        seen["user"] = str(msgs[-1].content)
        return "The total contract value is $2.4 million over three years [S1]."

    svc = make_service(structured_fn=router({"contract": {"query_type": "document", "document_query": "contract value"}}),
                       answer_fn=answer, min_relevance=0.0)
    svc.ingest_document("t1", "contract.pdf", injection_bytes)
    r = svc.chat("What is the contract value in the vendor contract?", "t1")
    assert "untrusted DATA" in seen["system"]
    assert 'warning="contains instruction-like text' in seen["user"]
    assert "2.4 million" in r.answer and r.verification_status == "VERIFIED"


def test_conversation_memory_reaches_the_model(make_service):
    prompts_seen = []

    def answer(msgs):
        prompts_seen.append([str(m.content) for m in msgs])
        return "Your favourite ticker is AMD." if len(prompts_seen) > 1 else "Nice to meet you, Priya!"

    svc = make_service(structured_fn=router({}), answer_fn=answer)
    svc.chat("My name is Priya and my favourite ticker is AMD.", "t1")
    r = svc.chat("What is my favourite ticker?", "t1")
    assert "AMD" in r.answer
    assert any("favourite ticker is AMD" in c for c in prompts_seen[1])


def test_clarification_for_finance_without_ticker(make_service):
    svc = make_service(structured_fn=router({"price": {"query_type": "finance"}}))
    r = svc.chat("What is the share price?", "t1")
    assert r.route == "clarification_required" and "ticker" in r.answer.lower()


def test_oversized_message_rejected_before_graph(make_service):
    from nova.errors import InputValidationError

    svc = make_service()
    with pytest.raises(InputValidationError):
        svc.chat("x" * 10_000, "t1")


# ------------------------------------------------------------- human-in-the-loop
def test_approval_interrupt_and_resume_approved(make_service):
    svc = make_service(answer_fn=grounded_answer({"333.69": "AAPL is at 333.69 USD [F1]."}))
    first = svc.chat("What is the current stock price of AAPL?", "t1", approval_required=True)
    assert first.status == "awaiting_approval"
    assert first.approval["operations"][0]["tool"] == "stock_quote"
    assert svc.get_thread("t1")["pending_approval"] is not None

    final = svc.resume("t1", approved=True)
    assert final.status == "completed" and "333.69" in final.answer
    assert "✓ External data approved" in final.timeline
    assert svc.pending_approval("t1") is None


def test_approval_rejected_skips_external_calls(make_service):
    def must_not_run(symbol):
        pytest.fail("external tool ran after rejection")

    svc = make_service(tools={"stock_quote": must_not_run})
    svc.chat("What is the current stock price of AAPL?", "t1", approval_required=True)
    final = svc.resume("t1", approved=False)
    assert final.status == "completed" and "did not run the external data operations" in final.answer


def test_new_message_blocked_while_approval_pending(make_service):
    from nova.errors import ApprovalStateError

    svc = make_service()
    svc.chat("What is the current stock price of AAPL?", "t1", approval_required=True)
    with pytest.raises(ApprovalStateError):
        svc.chat("hello", "t1")
    with pytest.raises(ApprovalStateError):
        svc.resume("other-thread", approved=True)
