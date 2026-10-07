"""
Calibrate NOVA_MIN_RELEVANCE on a DEV question set (disjoint from evaluation/dataset.json,
so the threshold is not tuned on the test set).

Prints the top-1 cosine similarity for answerable vs unanswerable dev questions, with
and without nomic task prefixes, and suggests the midpoint threshold.

    python -m evaluation.calibrate_threshold            # PDF threshold (NOVA_MIN_RELEVANCE)
    python -m evaluation.calibrate_threshold --video    # transcript threshold (NOVA_VIDEO_MIN_RELEVANCE)
"""

from __future__ import annotations

import os
import statistics
import tempfile

os.environ.setdefault("NOVA_ENV", "test")
os.environ.setdefault("NOVA_DATA_DIR", tempfile.mkdtemp(prefix="nova-calib-"))

from evaluation.fixtures.documents import fixture_path  # noqa: E402
from nova.config import get_settings  # noqa: E402
from nova.llm import get_embeddings  # noqa: E402
from nova.rag.ingest import parse_pdf  # noqa: E402
from nova.rag.retrieve import Retriever  # noqa: E402
from nova.rag.store import DocumentStore  # noqa: E402

DEV_ANSWERABLE = [
    "Where is Northwind headquartered?",
    "How much free cash flow did the company generate?",
    "What payload can the Atlas v3 carry?",
    "How many hospitals use the Helix system?",
    "Who audits the company's accounts?",
    "What share of revenue came from Europe?",
    "How much did SG&A expense amount to?",
    "What was the order backlog at year end?",
    "How many apprentices joined the apprenticeship programme?",
    "What operating margin does management expect next year?",
]
DEV_UNANSWERABLE = [
    "What is the company's dividend per share?",
    "Who is the Chief Financial Officer?",
    "What is the recipe for banana bread?",
    "How many stores does the company operate in Japan?",
    "What was the stock price at the IPO?",
    "What is the weather forecast for Pittsburgh?",
    "What is the capital of Australia?",
    "How tall is the Eiffel Tower?",
    "What is the company's credit rating?",
    "Which football team does the CEO support?",
]


def top_scores(prefixes: bool) -> tuple[list[float], list[float]]:
    s = get_settings()
    emb = get_embeddings(s, task_prefixes=prefixes)
    name = f"{s.embedding_model_name}{'+prefix' if prefixes else ''}"
    store = DocumentStore(s.vector_dir / name.replace(":", "_").replace("+", "_"), emb, name)
    data = fixture_path("northwind_annual_report_2025.pdf").read_bytes()
    store.add_document("calib", parse_pdf(data, "northwind_annual_report_2025.pdf"))
    retriever = Retriever(store, emb, min_relevance=-1.0)
    pos = [retriever.retrieve("calib", q).max_relevance for q in DEV_ANSWERABLE]
    neg = [retriever.retrieve("calib", q).max_relevance for q in DEV_UNANSWERABLE]
    return pos, neg


# Video dev set: questions about the transcript fixtures that are NOT in video_dataset.json.
VIDEO_DEV_ANSWERABLE = [
    ("quarterly_review_meeting", "Does the starter plan price change?"),
    ("quarterly_review_meeting", "Which trade shows did marketing spend its budget on?"),
    ("quarterly_review_meeting", "Who will share the detailed hiring budget?"),
    ("quarterly_review_meeting", "Which teams will the new engineers join?"),
    ("quarterly_review_meeting", "What is the issue with the hardware shipments?"),
    ("product_sync_meeting", "When will the Orbit mobile app launch?"),
    ("product_sync_meeting", "Which framework was chosen for the app?"),
    ("product_sync_meeting", "How many users will join the beta test?"),
    ("product_sync_meeting", "Why was the launch delayed?"),
    ("product_sync_meeting", "Will the app support offline mode?"),
]
VIDEO_DEV_UNANSWERABLE = [
    ("quarterly_review_meeting", "What did they say about the CEO's salary?"),
    ("quarterly_review_meeting", "Which bank provided the new loan?"),
    ("quarterly_review_meeting", "What was the weather like during the offsite?"),
    ("quarterly_review_meeting", "How many cybersecurity incidents happened?"),
    ("quarterly_review_meeting", "What is the recipe for biryani?"),
    ("product_sync_meeting", "What is the price of the premium subscription?"),
    ("product_sync_meeting", "Who won the cricket match?"),
    ("product_sync_meeting", "Which law firm reviewed the contract?"),
    ("product_sync_meeting", "How much revenue did the desktop app make last year?"),
    ("product_sync_meeting", "What is the capital of Australia?"),
]


def video_scores() -> tuple[list[float], list[float]]:
    from evaluation.fixtures.media import load_transcript, media_filename
    from nova.video.chunking import chunk_segments
    from nova.video.models import MediaAsset

    s = get_settings()
    emb = get_embeddings(s)
    store = DocumentStore(s.media_vector_dir / "calib", emb, s.embedding_model_name)
    for name in ("quarterly_review_meeting", "product_sync_meeting"):
        asset = MediaAsset(media_id=name.replace("_", "")[:24], thread_id="calib", filename=media_filename(name),
                           source_type="video", source_uri=media_filename(name))
        chunks = chunk_segments(load_transcript(name).segments, asset, s.video_chunk_size, s.video_chunk_overlap,
                                None, s.video_chunk_max_seconds)
        store.add_collection(name.replace("_", "-"), asset.media_id, chunks, {"media_id": asset.media_id})
    retriever = Retriever(store, emb, min_relevance=-1.0)
    pos = [retriever.retrieve(n.replace("_", "-"), q).max_relevance for n, q in VIDEO_DEV_ANSWERABLE]
    neg = [retriever.retrieve(n.replace("_", "-"), q).max_relevance for n, q in VIDEO_DEV_UNANSWERABLE]
    return pos, neg


def _print(label: str, pos: list[float], neg: list[float]) -> None:
    print(f"\n{label}")
    print(f"  answerable   top-1 cosine: min={min(pos):.3f} mean={statistics.mean(pos):.3f} max={max(pos):.3f}")
    print(f"  unanswerable top-1 cosine: min={min(neg):.3f} mean={statistics.mean(neg):.3f} max={max(neg):.3f}")
    print(f"  suggested threshold (midpoint of min(pos), max(neg)): {(min(pos) + max(neg)) / 2:.3f}")
    print(f"  separable: {min(pos) > max(neg)}")


def main() -> None:
    import sys

    if "--video" in sys.argv:
        pos, neg = video_scores()
        _print("video transcripts (task prefixes, current chunking)", pos, neg)
        for (_n, q), v in sorted(zip(VIDEO_DEV_ANSWERABLE, pos), key=lambda x: x[1]):
            print(f"  POS {v:.3f} {q}")
        for (_n, q), v in sorted(zip(VIDEO_DEV_UNANSWERABLE, neg), key=lambda x: x[1]):
            print(f"  NEG {v:.3f} {q}")
        return
    for prefixes in (False, True):
        pos, neg = top_scores(prefixes)
        print(f"\ntask prefixes={prefixes}")
        print(f"  answerable   top-1 cosine: min={min(pos):.3f} mean={statistics.mean(pos):.3f} max={max(pos):.3f}")
        print(f"  unanswerable top-1 cosine: min={min(neg):.3f} mean={statistics.mean(neg):.3f} max={max(neg):.3f}")
        print(f"  suggested threshold (midpoint of min(pos), max(neg)): {(min(pos) + max(neg)) / 2:.3f}")
        print(f"  separable: {min(pos) > max(neg)}")


if __name__ == "__main__":
    main()
