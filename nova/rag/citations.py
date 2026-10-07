"""
Source registry and citation validation over the unified evidence model (`nova.evidence`).

The model only ever sees short source ids: [S1] PDF chunk, [V1] video/audio transcript
window, [W1] web result, [F1] market data. After generation every marker is checked
against the registry: valid markers are rendered from *stored locators* —

    [PDF: strategy.pdf, p.12]   [VIDEO: meeting.mp4, 12:23–13:31]
    [WEB: [domain](url)]        [FINANCE: AAPL, Yahoo Finance, retrieved 2026-10-05T06:12:00+00:00]

— and unknown markers are removed and reported to the verifier. Citation text is never
taken from the model's imagination; model-written citations in the rendered style are
validated against the registry too.
"""

from __future__ import annotations

import re
from typing import Any

from nova.evidence import format_span, from_document_chunk, from_finance, from_video_chunk, from_web_result
from nova.safety import wrap_untrusted

_MARKER = re.compile(r"\[((?:[SVWF]\d{1,3})(?:\s*[,;]\s*[SVWF]\d{1,3})*)\]")
_SINGLE = re.compile(r"[SVWF]\d{1,3}")
# Model imitations of rendered styles (old "[Source: x, Page n]" and new "[PDF: x, p.n]").
_RENDERED_DOC = re.compile(r"\[(?:Source|PDF):\s*([^,\]]+?),\s*(?:Page|p\.)\s*(\d+)\]", re.I)
_RENDERED_VIDEO = re.compile(
    r"\[(?:VIDEO|AUDIO):\s*([^,\]]+?),\s*((?:\d+:)?\d{1,2}:\d{2})\s*[–-]\s*((?:\d+:)?\d{1,2}:\d{2})\]", re.I)


def _seconds(ts: str) -> float:
    parts = [int(p) for p in ts.split(":")]
    total = 0
    for p in parts:
        total = total * 60 + p
    return float(total)


def build_sources(
    chunks: list[dict[str, Any]],
    web_results: list[dict[str, Any]],
    finance_results: list[dict[str, Any]],
    video_chunks: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    sources = [from_document_chunk(i, c).as_source() for i, c in enumerate(chunks, start=1)]
    sources += [from_video_chunk(i, c).as_source() for i, c in enumerate(video_chunks or [], start=1)]
    sources += [from_web_result(i, w).as_source() for i, w in enumerate(web_results, start=1)]
    sources += [from_finance(i, f).as_source() for i, f in enumerate(finance_results, start=1)]
    return sources


def format_sources_for_prompt(sources: list[dict[str, Any]]) -> str:
    blocks = []
    for s in sources:
        if s["type"] == "document":
            header = f'type="pdf" file="{s["source_name"]}" page="{s["locator"]["page"]}"'
        elif s["type"] == "video":
            loc = s["locator"]
            span = format_span(loc["start_time"], loc["end_time"], s["metadata"].get("duration_seconds"))
            header = f'type="{s["source_type"]}_transcript" file="{s["source_name"]}" time="{span}"'
        elif s["type"] == "web":
            header = f'type="web" domain="{s["source_name"]}" title="{s["metadata"]["title"][:80]}"'
        else:
            header = f'type="market_data" label="{s["label"]}"'
        blocks.append(wrap_untrusted(s["id"], header, s["text"]))
    return "\n\n".join(blocks)


def render_citation(source: dict[str, Any]) -> str:
    kind, loc = source["type"], source["locator"]
    if kind == "document":
        return f"[PDF: {source['source_name']}, p.{loc['page']}]"
    if kind == "video":
        label = "AUDIO" if source["source_type"] == "audio" else "VIDEO"
        span = format_span(loc["start_time"], loc["end_time"], source["metadata"].get("duration_seconds"))
        return f"[{label}: {source['source_name']}, {span}]"
    if kind == "web":
        return f"[WEB: [{source['source_name']}]({loc['url']})]"
    retrieved = f", retrieved {loc['retrieved_at']}" if loc.get("retrieved_at") else ""
    return f"[FINANCE: {loc.get('symbol') or source['source_name']}, {source['source_name']}{retrieved}]"


def citation_record(source: dict[str, Any]) -> dict[str, Any]:
    rec: dict[str, Any] = {"id": source["id"], "type": source["type"], "source_type": source["source_type"],
                           "source_id": source["source_id"], "source_name": source["source_name"],
                           "locator": source["locator"], "relevance": source.get("relevance"),
                           "display": render_citation(source)}
    if source["type"] == "document":
        rec |= {"filename": source["source_name"], "page": source["locator"]["page"],
                "chunk_id": source["metadata"]["chunk_id"], "preview": source["text"][:300]}
    elif source["type"] == "video":
        rec |= {"media_id": source["source_id"], "start_time": source["locator"]["start_time"],
                "end_time": source["locator"]["end_time"], "preview": source["text"][:300]}
    elif source["type"] == "web":
        rec |= {"title": source["metadata"]["title"], "url": source["locator"]["url"], "domain": source["source_name"]}
    else:
        rec |= {"label": source["label"]}
    return rec


def apply_citations(text: str, sources: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]], list[str]]:
    """
    Validate and render citation markers.

    Returns (rendered_text, citations_used_in_order, invalid_markers).
    """
    by_id = {s["id"]: s for s in sources}
    used: dict[str, dict[str, Any]] = {}
    invalid: list[str] = []

    def _replace_marker(match: re.Match[str]) -> str:
        rendered: list[str] = []
        for sid in _SINGLE.findall(match.group(1)):
            src = by_id.get(sid)
            if src is None:
                invalid.append(sid)
                continue
            used.setdefault(sid, src)
            display = render_citation(src)
            if display not in rendered:
                rendered.append(display)
        return " ".join(rendered)

    text = _MARKER.sub(_replace_marker, text)

    def _check_rendered_doc(match: re.Match[str]) -> str:
        fname, page = match.group(1).strip(), int(match.group(2))
        for src in sources:
            if src["type"] == "document" and src["source_name"] == fname and src["locator"]["page"] == page:
                used.setdefault(src["id"], src)
                return render_citation(src)
        invalid.append(match.group(0))
        return ""

    def _check_rendered_video(match: re.Match[str]) -> str:
        name, start, end = match.group(1).strip(), _seconds(match.group(2)), _seconds(match.group(3))
        for src in sources:  # must overlap a time span that was actually retrieved
            loc = src["locator"]
            if (src["type"] == "video" and src["source_name"] == name
                    and start <= loc["end_time"] + 1 and end >= loc["start_time"] - 1):
                used.setdefault(src["id"], src)
                return render_citation(src)
        invalid.append(match.group(0))
        return ""

    text = _RENDERED_DOC.sub(_check_rendered_doc, text)
    text = _RENDERED_VIDEO.sub(_check_rendered_video, text)
    for display in {render_citation(s) for s in used.values()}:  # collapse adjacent duplicates
        d = re.escape(display)
        text = re.sub(rf"{d}(?:\s*[,;]?\s*{d})+", display.replace("\\", "\\\\"), text)
    text = re.sub(r"[ \t]+([.,;:])", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text).strip()
    return text, [citation_record(s) for s in used.values()], invalid
