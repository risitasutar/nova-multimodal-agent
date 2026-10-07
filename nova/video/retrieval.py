"""
Temporal retrieval over transcript chunks.

Reuses Nova's retriever (multi-query dense search + hybrid rerank + relevance floor)
against the media vector store. Every hit keeps its media id, time span and source name.
Scope: the store only returns collections owned by `thread_id`, and `media_ids` narrows
further — callers derive both from the agent's run config, never from user text.
"""

from __future__ import annotations

from typing import Any

from nova.rag.retrieve import RetrievalResult, Retriever


def search_media(
    retriever: Retriever,
    thread_id: str,
    query: str,
    rewritten: str | None = None,
    media_ids: list[str] | None = None,
    top_k: int | None = None,
    min_relevance: float | None = None,
) -> RetrievalResult:
    return retriever.retrieve(thread_id, query, rewritten, document_ids=media_ids or None, top_k=top_k,
                              min_relevance=min_relevance)


def to_results(result: RetrievalResult) -> list[dict[str, Any]]:
    """Structured evidence for API/tool output. `relevance` is the actual cosine similarity."""
    return [
        {
            "media_id": c["media_id"],
            "chunk_id": c["chunk_id"],
            "source_name": c["source_name"],
            "text": c["text"],
            "start_time": c["start_time"],
            "end_time": c["end_time"],
            "relevance": c["dense_score"],
            "rank": c.get("rank"),
        }
        for c in result.chunks
    ]
