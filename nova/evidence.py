"""
Unified evidence model shared by every knowledge source.

Whatever produced it — a PDF chunk, a transcript window, a web result, a market quote —
evidence reaching the model has the same shape:

    Evidence(id="V2", kind="video", source_type="video", source_id=<media_id>,
             source_name="meeting.mp4", content="...", relevance=0.71,
             locator={"start_time": 743.2, "end_time": 811.5}, metadata={...})

`locator` says *where* inside the source the content lives (page, time span, URL,
symbol + retrieval time). Citations are rendered from the locator, never from model text.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Kind = Literal["document", "video", "web", "finance"]
SourceType = Literal["pdf", "video", "audio", "youtube", "web", "finance"]
ID_PREFIX: dict[str, str] = {"document": "S", "video": "V", "web": "W", "finance": "F"}


class Evidence(BaseModel):
    id: str
    kind: Kind
    source_type: SourceType
    source_id: str
    source_name: str
    content: str
    locator: dict[str, Any] = Field(default_factory=dict)
    relevance: float | None = None  # actual retrieval score when one exists; never invented
    metadata: dict[str, Any] = Field(default_factory=dict)

    def as_source(self) -> dict[str, Any]:
        """Flat dict stored in graph state (JSON-serialisable; legacy keys kept for callers)."""
        base = self.model_dump()
        base["type"] = self.kind
        base["text"] = self.content
        base.update(self.metadata.get("legacy", {}))
        return base


def format_timestamp(seconds: float, long_form: bool = False) -> str:
    """MM:SS for media under an hour, H:MM:SS otherwise (long_form forces hours)."""
    total = max(0, int(round(seconds)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if (long_form or h) else f"{m:02d}:{s:02d}"


def format_span(start: float, end: float, duration: float | None = None) -> str:
    long_form = bool(duration and duration >= 3600) or end >= 3600
    return f"{format_timestamp(start, long_form)}–{format_timestamp(end, long_form)}"


# ---------------------------------------------------------------- constructors
def from_document_chunk(i: int, c: dict[str, Any]) -> Evidence:
    return Evidence(
        id=f"S{i}", kind="document", source_type="pdf", source_id=c["document_id"], source_name=c["filename"],
        content=c["text"], locator={"page": c["page"]}, relevance=c.get("dense_score"),
        metadata={"chunk_id": c["chunk_id"], "injection_flags": c.get("injection_flags", []),
                  "legacy": {"filename": c["filename"], "page": c["page"], "chunk_id": c["chunk_id"],
                             "document_id": c["document_id"], "score": c.get("dense_score"),
                             "injection_flags": c.get("injection_flags", [])}},
    )


def from_video_chunk(i: int, c: dict[str, Any]) -> Evidence:
    locator = {"start_time": float(c["start_time"]), "end_time": float(c["end_time"])}
    return Evidence(
        id=f"V{i}", kind="video", source_type=c.get("source_type", "video"), source_id=c["media_id"],
        source_name=c["source_name"], content=c["text"], locator=locator, relevance=c.get("dense_score"),
        metadata={"chunk_id": c.get("chunk_id"), "duration_seconds": c.get("duration_seconds"),
                  "insight": c.get("insight"), "injection_flags": c.get("injection_flags", []),
                  "legacy": {"media_id": c["media_id"], "chunk_id": c.get("chunk_id"), **locator,
                             "injection_flags": c.get("injection_flags", [])}},
    )


def from_web_result(i: int, w: dict[str, Any]) -> Evidence:
    return Evidence(
        id=f"W{i}", kind="web", source_type="web", source_id=w["url"], source_name=w["domain"],
        content=w.get("snippet", ""), locator={"url": w["url"], "retrieved_at": w.get("retrieved_at")},
        metadata={"title": w["title"], "date": w.get("date"),
                  "legacy": {"title": w["title"], "url": w["url"], "domain": w["domain"], "date": w.get("date")}},
    )


def from_finance(i: int, f: dict[str, Any]) -> Evidence:
    return Evidence(
        id=f"F{i}", kind="finance", source_type="finance", source_id=f.get("symbol", ""),
        source_name=f.get("provider") or f["label"], content=f["text"],
        locator={"symbol": f.get("symbol"), "metric": f.get("metric"), "retrieved_at": f.get("retrieved_at")},
        metadata={"legacy": {"label": f["label"]}},
    )
