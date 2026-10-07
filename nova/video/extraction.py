"""
Verification of extracted meeting insights against the transcript.

The model proposes items citing transcript chunk ids. This module keeps only what the
transcript supports:
  * items citing no valid chunk id are aligned to the chunk whose words support them
    (≥60% of the item's content words); if no chunk supports them they are dropped;
  * an owner is kept only if that name occurs in the cited chunk text;
  * a deadline is kept only if all of its words occur in the cited chunk text;
  * timestamps come from the cited chunks, never from the model.
Unsupported owners/deadlines become None and are shown as "Not identified in the transcript."
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from nova.observability import log_event
from nova.video.models import InsightItem

NOT_IDENTIFIED = "Not identified in the transcript."
_WORD = re.compile(r"[a-z0-9]+")
_EMPTY = {"", "none", "n/a", "na", "null", "unknown", "not specified", "not mentioned", "tbd", "unassigned",
          "not identified", "not identified in the transcript", "not identified in the transcript."}


class RawItem(BaseModel):
    text: str
    chunk_ids: list[str] = Field(description="ids like C3 of the transcript lines that support this item")


class RawAction(BaseModel):
    task: str
    owner: str | None = None
    deadline: str | None = None
    chunk_ids: list[str] = Field(description="ids like C3 of the transcript lines that support this item")


class RawInsights(BaseModel):
    """Structured output requested from the LLM for one transcript window."""

    title: str | None = None
    summary_points: list[RawItem] = Field(default_factory=list)
    decisions: list[RawItem] = Field(default_factory=list)
    action_items: list[RawAction] = Field(default_factory=list)
    open_questions: list[RawItem] = Field(default_factory=list)


_STOP = {"the", "and", "for", "with", "that", "this", "will", "from", "have", "has", "was", "were", "are", "our",
         "their", "they", "them", "about", "into", "need", "needs", "should", "would", "could", "whether"}


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def align(text: str, chunks_by_id: dict[str, dict[str, Any]], min_overlap: float = 0.6) -> str | None:
    """Chunk id whose text contains most of the item's content words (None if none qualifies)."""
    words = {w for w in _words(text) if len(w) > 2 and w not in _STOP}
    if not words:
        return None
    best_id, best = None, 0.0
    for cid, chunk in chunks_by_id.items():
        score = len(words & _words(chunk["text"])) / len(words)
        if score > best:
            best_id, best = cid, score
    return best_id if best >= min_overlap else None


def _supported(value: str | None, evidence: str) -> str | None:
    if value is None or value.strip().lower() in _EMPTY:
        return None
    needed = _words(value)
    return value.strip() if needed and needed <= _words(evidence) else None


def verify_items(
    raw: list[Any], chunks_by_id: dict[str, dict[str, Any]], *, action: bool = False
) -> tuple[list[InsightItem], int]:
    """Return (verified items, number dropped)."""
    out: list[InsightItem] = []
    dropped = 0
    seen: set[str] = set()
    for item in raw:
        ids = [cid for cid in dict.fromkeys(item.chunk_ids) if cid in chunks_by_id]
        text = (item.task if action else item.text).strip()
        key = text.lower()
        if text and not ids:
            aligned = align(text, chunks_by_id)
            ids = [aligned] if aligned else []
        if not ids or not text or key in seen:
            dropped += 1
            log_event("insight_item_dropped", level=20, text=text[:120],
                      reason="duplicate" if key in seen else "no supporting transcript chunk")
            continue
        seen.add(key)
        cited = [chunks_by_id[cid] for cid in ids]
        evidence = " ".join(c["text"] for c in cited)
        out.append(
            InsightItem(
                text=text,
                owner=_supported(getattr(item, "owner", None), evidence) if action else None,
                deadline=_supported(getattr(item, "deadline", None), evidence) if action else None,
                chunk_ids=ids,
                start_time=min(c["start_time"] for c in cited),
                end_time=max(c["end_time"] for c in cited),
            )
        )
    return out, dropped
