# ADR-002: FAISS vector store without pickle

**Status:** Accepted · 2026-10-05

## Context
Upstream persisted LangChain FAISS indexes with `save_local` and reloaded them with
`load_local(..., allow_dangerous_deserialization=True)`, which unpickles `index.pkl`. Anyone able to write to
`data/vectorstores/` could get code execution. One index per thread also meant a second upload overwrote the first.

## Decision
Keep FAISS: it runs in-process, needs no server, and is fast for per-conversation corpora. Nova owns the
persistence format (`nova/rag/store.py`):

| File | Content |
|---|---|
| `index.faiss` | Raw FAISS binary. `IndexFlatIP` over L2-normalised vectors gives cosine similarity, so scores are interpretable for the relevance threshold. |
| `chunks.json` | Chunk text and metadata as plain JSON. |
| `manifest.json` | Thread id, document id, embedding model, dimensions, and the SHA-256 of both files. |

The layout is `vectorstores/<thread_id>/<document_id>/`, so a thread holds many documents and each can be
deleted on its own.

**Trust boundary:** an index is loaded only if all of these hold:
- the ids are well-formed (no path traversal) and resolve inside the vector root;
- the manifest names the same thread and document as the directory;
- the embedding model matches the current one;
- both checksums match the manifest.

Nothing is ever unpickled.

## Consequences
- Tampered, foreign or stale-model indexes are rejected and logged (`tests/rag/test_store_retrieval.py`).
- Flat search costs O(n) per query. That is fine for documents with thousands of chunks; a large shared corpus
  would need an ANN index (IVF/HNSW) or a vector database (pgvector, Qdrant).
- Pickle indexes from the pre-upgrade app are ignored, so those PDFs need re-uploading.
