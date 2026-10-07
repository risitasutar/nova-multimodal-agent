"""
Cross-modal reasoning with deterministic fixtures:
    meeting (video) says the Q4 target is 10 crore; strategy PDF says 12 crore.
Expected: inconsistent; difference 2 crore; percentage difference defined as
(strategy − meeting) / meeting × 100 = 20 %.
"""

import re

from evaluation.fixtures.media import strategy_pdf
from tests.video.helpers import add_fixture_media, structured

_SRC = re.compile(r'<source id="([SV]\d+)"[^>]*>\n(.*?)\n</source>', re.S)


def _answer_from_prompt(msgs):
    prompt = str(msgs[-1].content)
    srcs = dict(_SRC.findall(prompt))
    vid = next(k for k, v in srcs.items() if k.startswith("V") and "10 crore" in v)
    doc = next(k for k, v in srcs.items() if k.startswith("S") and "12 crore" in v)
    calc = re.search(r"pct: \(12 - 10\) / 10 \* 100 = ([\d.]+)", prompt)
    pct = calc.group(1) if calc else "?"
    return (f"**Video evidence:** the meeting set the Q4 target at 10 crore [{vid}].\n\n"
            f"**Document evidence:** the strategy sets it at 12 crore [{doc}].\n\n"
            f"**Comparison:** the targets are not consistent: a difference of 2 crore, {pct}% above the meeting target.")


def _service(make_service):
    svc = make_service(structured_fn=structured({
        "consistent": {"query_type": "multi_tool", "media_query": "Q4 revenue target",
                       "document_query": "Q4 revenue target"},
        "__plan__": {"steps": [{"tool": "video_search", "input": "Q4 revenue target"},
                               {"tool": "document_search", "input": "Q4 revenue target"}],
                     "needs_calculation": True},
        "_CalculationPlan": {"calculations": [{"label": "difference", "expression": "12 - 10"},
                                              {"label": "pct", "expression": "(12 - 10) / 10 * 100"}]},
    }), answer_fn=_answer_from_prompt, min_relevance=0.0, video_min_relevance=0.0)
    svc.ingest_document("x1", "strategy_2026.pdf", strategy_pdf())
    add_fixture_media(svc, "x1", insights=False)
    return svc


def test_video_pdf_calculator_and_citations(make_service):
    svc = _service(make_service)
    r = svc.chat("Is the Q4 revenue target in the meeting consistent with the strategy document?", "x1")
    assert r.route == "multi_tool"
    assert {"video_search", "document_search", "calculator"} <= set(r.tools_used)
    assert "not consistent" in r.answer and "2 crore" in r.answer and "20%" in r.answer
    kinds = {c["type"] for c in r.citations}
    assert kinds == {"video", "document"}
    video = next(c for c in r.citations if c["type"] == "video")
    assert video["start_time"] <= 743.2 <= video["end_time"]  # the real time the target was set
    doc = next(c for c in r.citations if c["type"] == "document")
    assert doc["page"] == 1
    assert r.verification_status == "VERIFIED"
    assert any("Calculation completed (12 - 10)" in s for s in r.timeline)


def test_cross_modal_evidence_stays_in_thread(make_service):
    svc = _service(make_service)
    svc.deps.llm.answer_fn = lambda msgs: "The sources do not contain this information."
    other = svc.chat("Is the Q4 revenue target in the meeting consistent with the strategy document?", "other-thread")
    assert not other.citations  # neither the PDF nor the meeting of thread x1 leaks into another thread
