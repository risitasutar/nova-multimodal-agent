"""
Retrieval: multi-query dense search → hybrid lexical rerank → relevance filter.

* Query rewrite: the classifier produces a standalone `document_query` (resolving
  follow-ups like "what about its risks?"). Both the raw question and the rewrite are
  searched and merged, so a bad rewrite cannot hide a good raw match.
* Rerank: 0.75·cosine + 0.25·lexical overlap. Cheap, deterministic, explainable; it
  rescues exact-term matches (numbers, names) that dense embeddings blur.
* Relevance filter: chunks whose cosine similarity is below `min_relevance` are not
  evidence. If nothing survives, the agent answers "insufficient evidence".
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.embeddings import Embeddings

from nova.rag.store import DocumentStore

_STOPWORDS = frozenset(
    "a an and are as at be by did do does for from has have how in is it its of on or per "
    "that the their this to was were what when where which who why will with according "
    "document report pdf uploaded file my me tell about please say says said".split()
)
_TOKEN = re.compile(r"[a-z0-9][a-z0-9.%\-]*")
DENSE_WEIGHT = 0.75


def tokenize(text: str) -> list[str]:
    return [t.strip(".-") for t in _TOKEN.findall(text.lower()) if t.strip(".-") and t not in _STOPWORDS]


def lexical_overlap(query: str, text: str) -> float:
    q_terms = {t for t in tokenize(query) if len(t) > 2 or t.isdigit()}
    if not q_terms:
        return 0.0
    doc_terms = set(tokenize(text))
    return len(q_terms & doc_terms) / len(q_terms)


@dataclass
class RetrievalResult:
    chunks: list[dict[str, Any]] = field(default_factory=list)
    candidates: int = 0
    max_relevance: float = 0.0
    queries: list[str] = field(default_factory=list)
    latency_ms: float = 0.0

    @property
    def sufficient(self) -> bool:
        return bool(self.chunks)


class Retriever:
    def __init__(
        self,
        store: DocumentStore,
        embeddings: Embeddings,
        *,
        top_k: int = 4,
        candidates: int = 12,
        min_relevance: float = 0.55,
    ) -> None:
        self.store = store
        self.embeddings = embeddings
        self.top_k = top_k
        self.candidates = candidates
        self.min_relevance = min_relevance

    def retrieve(
        self,
        thread_id: str,
        query: str,
        rewritten: str | None = None,
        *,
        document_ids: list[str] | None = None,
        top_k: int | None = None,
        min_relevance: float | None = None,
    ) -> RetrievalResult:
        start = time.perf_counter()
        k = top_k or self.top_k
        floor = self.min_relevance if min_relevance is None else min_relevance
        queries = [query]
        if rewritten and rewritten.strip().lower() != query.strip().lower():
            queries.append(rewritten)

        merged: dict[str, tuple[dict[str, Any], float]] = {}
        for q in queries:
            vector = self.embeddings.embed_query(q)
            for chunk, score in self.store.search(thread_id, vector, self.candidates, document_ids):
                prev = merged.get(chunk["chunk_id"])
                if prev is None or score > prev[1]:
                    merged[chunk["chunk_id"]] = (chunk, score)

        lexical_query = " ".join(queries)
        scored = []
        for chunk, dense in merged.values():
            lexical = lexical_overlap(lexical_query, chunk["text"])
            scored.append(
                {
                    **chunk,
                    "dense_score": round(dense, 4),
                    "lexical_score": round(lexical, 4),
                    "score": round(DENSE_WEIGHT * dense + (1 - DENSE_WEIGHT) * lexical, 4),
                }
            )
        scored.sort(key=lambda c: c["score"], reverse=True)
        kept = [c for c in scored if c["dense_score"] >= floor][:k]
        for rank, chunk in enumerate(kept, start=1):
            chunk["rank"] = rank
        return RetrievalResult(
            chunks=kept,
            candidates=len(scored),
            max_relevance=max((c["dense_score"] for c in scored), default=0.0),
            queries=queries,
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
        )
