from nova.evidence import format_span, format_timestamp
from nova.rag.citations import apply_citations, build_sources, format_sources_for_prompt

CHUNKS = [
    {"chunk_id": "c3", "document_id": "d1", "filename": "report.pdf", "page": 3, "text": "Net income was $28.1 million.",
     "dense_score": 0.8},
    {"chunk_id": "c7", "document_id": "d1", "filename": "report.pdf", "page": 7, "text": "Kestrel supplies chips.",
     "dense_score": 0.7, "injection_flags": []},
]
VIDEO = [
    {"chunk_id": "m1-c4", "media_id": "m1", "source_name": "meeting.mp4", "source_type": "video",
     "text": "We agreed to raise the Q4 target to 10 crore.", "start_time": 743.2, "end_time": 811.5,
     "duration_seconds": 1800.0, "dense_score": 0.74},
]
WEB = [{"title": "Eiffel Tower", "url": "https://en.wikipedia.org/wiki/Eiffel_Tower", "domain": "en.wikipedia.org",
        "snippet": "Completed in 1889."}]
FIN = [{"label": "Yahoo Finance (AAPL quote, 2026-10-02)", "text": "AAPL latest price 333.69 USD.", "symbol": "AAPL",
        "provider": "Yahoo Finance", "retrieved_at": "2026-10-05T06:12:00+00:00"}]


def sources():
    return build_sources(CHUNKS, WEB, FIN, VIDEO)


def test_unified_evidence_shape_and_ids():
    srcs = sources()
    assert [s["id"] for s in srcs] == ["S1", "S2", "V1", "W1", "F1"]
    for s in srcs:  # one evidence schema across all source kinds
        assert {"id", "kind", "source_type", "source_id", "source_name", "content", "locator", "relevance"} <= set(s)
    assert srcs[0]["locator"] == {"page": 3} and srcs[0]["relevance"] == 0.8
    assert srcs[2]["locator"] == {"start_time": 743.2, "end_time": 811.5} and srcs[2]["source_id"] == "m1"
    assert srcs[3]["locator"] == {"url": WEB[0]["url"], "retrieved_at": None} and srcs[3]["relevance"] is None  # no invented score
    assert srcs[4]["locator"] == {"symbol": "AAPL", "metric": None, "retrieved_at": "2026-10-05T06:12:00+00:00"}


def test_prompt_format():
    prompt = format_sources_for_prompt(sources())
    assert '<source id="S1" type="pdf" file="report.pdf" page="3">' in prompt
    assert '<source id="V1" type="video_transcript" file="meeting.mp4" time="12:23–13:32">' in prompt
    assert '<source id="W1" type="web"' in prompt and '<source id="F1" type="market_data"' in prompt


def test_valid_markers_render_from_stored_metadata():
    text, used, invalid = apply_citations("Net income was $28.1 million [S1].", sources())
    assert text == "Net income was $28.1 million [PDF: report.pdf, p.3]."
    assert invalid == [] and used[0]["page"] == 3 and used[0]["chunk_id"] == "c3"


def test_video_marker_renders_timestamps():
    text, used, invalid = apply_citations("The target is 10 crore [V1].", sources())
    assert text == "The target is 10 crore [VIDEO: meeting.mp4, 12:23–13:32]."
    assert used[0]["type"] == "video" and used[0]["start_time"] == 743.2 and used[0]["media_id"] == "m1"


def test_invalid_markers_removed_and_reported():
    text, used, invalid = apply_citations("A fact [S9]. Another [S2]. Video [V4].", sources())
    assert "[S9]" not in text and "[V4]" not in text and invalid == ["S9", "V4"]
    assert [u["id"] for u in used] == ["S2"]


def test_grouped_and_mixed_markers():
    text, used, _ = apply_citations("Facts [S1, S2] and [W1][F1] and [V1].", sources())
    assert "[PDF: report.pdf, p.3] [PDF: report.pdf, p.7]" in text
    assert "[WEB: [en.wikipedia.org](https://en.wikipedia.org/wiki/Eiffel_Tower)]" in text
    assert "[FINANCE: AAPL, Yahoo Finance, retrieved 2026-10-05T06:12:00+00:00]" in text
    assert {u["type"] for u in used} == {"document", "web", "finance", "video"}


def test_model_written_rendered_citations_are_validated():
    ok, used, invalid = apply_citations("X [PDF: report.pdf, p.7]. Y [Source: report.pdf, Page 3].", sources())
    assert invalid == [] and [u["page"] for u in used] == [7, 3]
    bad, used, invalid = apply_citations("X [PDF: report.pdf, p.40].", sources())
    assert "p.40" not in bad and invalid and not used


def test_model_written_video_citation_must_overlap_retrieved_span():
    ok, used, invalid = apply_citations("Agreed [VIDEO: meeting.mp4, 12:30–12:50].", sources())
    assert invalid == [] and used[0]["id"] == "V1" and "12:23–13:32" in ok
    bad, used, invalid = apply_citations("Agreed [VIDEO: meeting.mp4, 25:00–26:00].", sources())
    assert "25:00" not in bad and invalid and not used


def test_duplicate_adjacent_citations_collapse():
    text, _, _ = apply_citations("Fact [S1] [S1].", sources())
    assert text.count("p.3") == 1


def test_timestamp_formatting():
    assert format_timestamp(743.2) == "12:23"
    assert format_timestamp(3725) == "1:02:05"
    assert format_span(5, 65) == "00:05–01:05"
    assert format_span(5, 65, duration=4000) == "0:00:05–0:01:05"


def test_comma_separated_duplicate_citations_collapse():
    srcs = build_sources(CHUNKS, [], [], VIDEO + [{**VIDEO[0], "chunk_id": "m1-insight"}])
    text, _, _ = apply_citations("Target is 10 crore [V1], [V2].", srcs)
    assert text.count("[VIDEO: meeting.mp4, 12:23–13:32]") == 1
