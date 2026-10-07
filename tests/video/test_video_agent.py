"""The LangGraph agent with the video capability: routing, temporal citations, insights, memory, safety."""

import re

import pytest

from nova.agent import prompts
from tests.video.helpers import add_fixture_media, structured

_SOURCE = re.compile(r'<source id="(V\d+)"[^>]*>\n(.*?)\n</source>', re.S)


def cite_first_video_source_containing(*needles, sentence="Answer"):
    """Scripted answer: cite the first [V#] source whose text contains any needle."""

    def _fn(msgs):
        prompt = str(msgs[-1].content)
        for sid, body in _SOURCE.findall(prompt):
            if any(n.lower() in body.lower() for n in needles):
                return f"{sentence} [{sid}]."
        return "The sources do not contain this information."

    return _fn


def test_temporal_question_routes_to_video_and_cites_timestamps(make_service):
    svc = make_service(structured_fn=structured(),
                       answer_fn=cite_first_video_source_containing("hiring plan", sentence="The hiring plan was discussed"))
    add_fixture_media(svc, "t1")
    r = svc.chat("When did they discuss the hiring plan?", "t1")
    assert r.route == "video" and r.tools_used == ["video_search"]
    assert r.verification_status == "VERIFIED"
    assert "[VIDEO: quarterly_review_meeting.mp4, " in r.answer and "[V1]" not in r.answer
    cite = r.citations[0]
    assert cite["type"] == "video" and cite["start_time"] <= 550.0 <= cite["end_time"]
    assert any("Retrieved meeting evidence" in step for step in r.timeline)


def test_decisions_come_from_verified_insights_with_timestamps(make_service):
    svc = make_service(structured_fn=structured(),
                       answer_fn=cite_first_video_source_containing("Key decision", sentence="They raised the target"))
    add_fixture_media(svc, "t1")
    r = svc.chat("What were the key decisions in the meeting?", "t1")
    assert "media_insights" in r.tools_used
    assert r.citations and r.citations[0]["type"] == "video" and r.citations[0]["end_time"] > 0
    assert any("Meeting insights loaded" in step for step in r.timeline)


def test_unanswerable_video_question_is_not_answered(make_service):
    svc = make_service(structured_fn=structured(), answer_fn=lambda m: pytest.fail("must not answer"),
                       video_min_relevance=0.95)
    add_fixture_media(svc, "t1")
    r = svc.chat("What did they say in the meeting about the dividend payout?", "t1")
    assert r.answer == prompts.INSUFFICIENT_MEDIA_EVIDENCE and r.verification_status == "INSUFFICIENT_EVIDENCE"


def test_media_question_without_media(make_service):
    svc = make_service(structured_fn=structured({"meeting": {"query_type": "video"}}))
    r = svc.chat("Summarize this meeting.", "t1")
    assert r.answer == prompts.NO_MEDIA_UPLOADED


def test_transcript_injection_is_treated_as_data(make_service):
    seen = {}

    def answer(msgs):
        seen["system"], seen["user"] = str(msgs[0].content), str(msgs[-1].content)
        sid = next(s for s, body in _SOURCE.findall(seen["user"]) if "25 lakh" in body)
        return f"The approved budget is 25 lakh rupees per year [{sid}]."

    svc = make_service(structured_fn=structured(), answer_fn=answer, video_min_relevance=0.0)
    add_fixture_media(svc, "t1", "vendor_call_injection", insights=False)
    r = svc.chat("What budget was approved in the vendor call?", "t1")
    assert "untrusted DATA" in seen["system"]
    assert 'warning="contains instruction-like text' in seen["user"]
    assert "25 lakh" in r.answer and "[AUDIO: vendor_call_injection.mp3" in r.answer
    assert r.verification_status == "VERIFIED"


def test_ambiguous_meeting_reference_asks_then_memory_resolves(make_service):
    svc = make_service(
        structured_fn=structured({"approve": {"query_type": "video", "media_query": "approved budget"}}),
        answer_fn=cite_first_video_source_containing("budget", "Flutter", sentence="Answer"),
        video_min_relevance=0.0,
    )
    add_fixture_media(svc, "t1", "quarterly_review_meeting", insights=False)
    add_fixture_media(svc, "t1", "product_sync_meeting", insights=False)
    first = svc.chat("What was decided in the meeting?", "t1")
    assert first.route == "clarification_required"
    assert "quarterly_review_meeting.mp4" in first.answer and "product_sync_meeting.mp4" in first.answer

    named = svc.chat("What was decided in the product sync meeting?", "t1")
    assert named.route == "video" and all("product_sync" in c["source_name"] for c in named.citations)

    # Follow-up with a pronoun: resolved to the recording discussed in the previous turn.
    follow = svc.chat("What budget did they approve?", "t1")
    assert follow.route == "video"
    assert follow.citations and all(c["source_name"] == "product_sync_meeting.mp4" for c in follow.citations)


def test_compare_all_meetings_is_not_ambiguous(make_service):
    svc = make_service(structured_fn=structured(), answer_fn=cite_first_video_source_containing("budget"),
                       video_min_relevance=0.0)
    add_fixture_media(svc, "t1", "quarterly_review_meeting", insights=False)
    add_fixture_media(svc, "t1", "product_sync_meeting", insights=False)
    r = svc.chat("Compare the budgets discussed across both meetings.", "t1")
    assert r.route != "clarification_required" and "video_search" in r.tools_used


def test_video_plus_calculator(make_service):
    def answer(msgs):
        prompt = str(msgs[-1].content)
        sid = next(s for s, body in _SOURCE.findall(prompt) if "10 crore" in body)
        return f"The target is 10 crore [{sid}]; reaching it from 8 crore needs a 25% increase."

    svc = make_service(
        structured_fn=structured({"_CalculationPlan": {"calculations": [{"label": "increase %", "expression": "(10 - 8) / 8 * 100"}]}}),
        answer_fn=answer, video_min_relevance=0.0)
    add_fixture_media(svc, "t1", insights=False)
    r = svc.chat("The meeting set a revenue target. Current revenue is 8 crore. What percentage increase is required?", "t1")
    assert {"video_search", "calculator"} <= set(r.tools_used)
    assert r.verification_status == "VERIFIED" and "25%" in r.answer


def test_video_plus_finance(make_service):
    def answer(msgs):
        prompt = str(msgs[-1].content)
        vid = next(s for s, body in _SOURCE.findall(prompt) if "crore" in body)
        return f"AAPL trades at 333.69 USD [F1]; the meeting target was 10 crore [{vid}]."

    svc = make_service(structured_fn=structured({
        "apple": {"query_type": "multi_tool", "tickers": ["AAPL"], "finance_needs": ["quote"],
                  "media_query": "revenue target"},
        "__plan__": {"steps": [{"tool": "stock_quote", "input": "AAPL"}]},
    }), answer_fn=answer, video_min_relevance=0.0)
    add_fixture_media(svc, "t1", insights=False)
    r = svc.chat("Compare Apple's current stock price with the revenue target discussed in the meeting.", "t1")
    # the router's video requirement is enforced even though the model plan omitted it
    assert {"stock_quote", "video_search"} <= set(r.tools_used)
    assert {c["type"] for c in r.citations} == {"finance", "video"}


def test_cross_modal_prompt_separates_video_and_document(make_service):
    from evaluation.fixtures.media import strategy_pdf

    seen = {}

    def answer(msgs):
        seen["user"] = str(msgs[-1].content)
        srcs = dict(_SOURCE.findall(seen["user"]))
        doc = re.search(r'<source id="(S\d+)"[^>]*>\n[^<]*12 crore', seen["user"]).group(1)
        vid = next(s for s, b in srcs.items() if "10 crore" in b)
        return (f"**Video evidence:** target 10 crore [{vid}].\n\n**Document evidence:** target 12 crore [{doc}].\n\n"
                "**Comparison:** inconsistent; the strategy is 2 crore (20%) higher.")

    svc = make_service(structured_fn=structured({
        "consistent": {"query_type": "multi_tool", "media_query": "revenue target", "document_query": "revenue target"},
        "__plan__": {"steps": [{"tool": "video_search", "input": "revenue target"},
                               {"tool": "document_search", "input": "revenue target"}], "needs_calculation": True},
        "_CalculationPlan": {"calculations": [{"label": "difference", "expression": "12 - 10"},
                                              {"label": "pct", "expression": "(12 - 10) / 10 * 100"}]},
    }), answer_fn=answer, min_relevance=0.0, video_min_relevance=0.0)
    svc.ingest_document("t1", "strategy_2026.pdf", strategy_pdf())
    add_fixture_media(svc, "t1", insights=False)
    r = svc.chat("Is the revenue target in the meeting consistent with the strategy document?", "t1")
    assert prompts.CROSS_SOURCE_NOTE in seen["user"]
    assert {"video_search", "document_search", "calculator"} <= set(r.tools_used)
    kinds = {c["type"] for c in r.citations}
    assert kinds == {"video", "document"} and "[PDF: strategy_2026.pdf, p.1]" in r.answer
    assert "[VIDEO: quarterly_review_meeting.mp4," in r.answer and r.verification_status == "VERIFIED"


def test_router_examples():
    from nova.agent.classifier import heuristic_classify

    def route(q, docs=True, media=True):
        return heuristic_classify(q, docs, media)

    assert route("Summarize this meeting.").query_type == "video"
    assert route("Summarize this meeting.").insight_request == ["summary"]
    assert route("What did they decide about pricing?").query_type == "video"
    assert route("When did they discuss the hiring plan?").query_type == "video"
    assert route("What does the uploaded PDF say about pricing?").query_type == "document"
    assert route("Compare the meeting decision with the PDF.").query_type == "multi_tool"
    assert route("What is Apple's current price?", docs=False, media=False).query_type == "finance"
    assert route("Search the latest news about Apple.", docs=False, media=False).query_type == "finance"
    assert route("Compare Apple's current performance with what was discussed in the meeting.").query_type == "multi_tool"
    assert route("What are the action items?").insight_request == ["action_items"]
