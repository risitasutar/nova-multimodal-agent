"""
Per-thread, per-document FAISS store — without pickle.

Layout:  <vector_dir>/<thread_id>/<document_id>/
             index.faiss    raw FAISS binary (inner product over L2-normalised vectors = cosine)
             chunks.json    chunk texts + metadata (plain JSON)
             manifest.json  ownership + integrity record (thread, document, model, SHA-256s)

Trust boundary (see docs/adr/ADR-002): an index is only loaded when
  * the thread/document ids are well-formed (no path traversal) and the path resolves
    inside the configured vector directory,
  * the manifest names the same thread and document as the directory,
  * the embedding model matches the current one, and
  * the SHA-256 of index.faiss and chunks.json match the manifest.
No pickle is ever deserialised, so a tampered file cannot execute code.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from langchain_core.embeddings import Embeddings

from nova.errors import DocumentError, IndexIntegrityError
from nova.observability import log_event
from nova.rag.ingest import ParsedDocument

FORMAT_VERSION = 1
_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
_EMBED_BATCH = 32


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype("float32")


class _LoadedIndex:
    def __init__(self, index: faiss.Index, chunks: list[dict], manifest: dict) -> None:
        self.index = index
        self.chunks = chunks
        self.manifest = manifest


class DocumentStore:
    def __init__(self, root: Path, embeddings: Embeddings, embedding_model: str) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.embeddings = embeddings
        self.embedding_model = embedding_model
        self._cache: dict[tuple[str, str], _LoadedIndex] = {}
        self._lock = threading.RLock()

    # ---------------- paths / trust boundary ----------------
    def _safe_dir(self, *parts: str) -> Path:
        for part in parts:
            if not _ID_RE.match(part or ""):
                raise DocumentError(f"invalid id {part!r}", user_message="Invalid conversation or document id.")
        path = self.root.joinpath(*parts).resolve()
        if self.root not in path.parents and path != self.root:
            raise DocumentError("path escapes vector root")
        return path

    # ---------------- write ----------------
    def add_document(self, thread_id: str, doc: ParsedDocument) -> dict:
        info = self.add_collection(
            thread_id,
            doc.document_id,
            [c.to_dict() for c in doc.chunks],
            {"document_id": doc.document_id, "filename": doc.filename, "file_sha256": doc.file_sha256,
             "pages": doc.pages},
        )
        log_event("document_indexed", document_id=doc.document_id, chunks=len(doc.chunks), pages=doc.pages)
        return info

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), _EMBED_BATCH):
            vectors.extend(self.embeddings.embed_documents(texts[i : i + _EMBED_BATCH]))
        return _normalise(np.asarray(vectors, dtype="float32"))

    def add_collection(
        self,
        thread_id: str,
        collection_id: str,
        chunks: list[dict],
        fields: dict,
        vectors: np.ndarray | None = None,
    ) -> dict:
        """
        Index arbitrary chunk dicts (each needs `chunk_id` and `text`) as one collection
        owned by `thread_id`. `fields` are stored in the manifest and returned by listings.
        Pre-computed `vectors` (e.g. from the media embedding cache) skip re-embedding.
        """
        matrix = self.embed_texts([c["text"] for c in chunks]) if vectors is None else _normalise(vectors)
        if matrix.shape[0] != len(chunks):
            raise ValueError("vector/chunk count mismatch")
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)

        with self._lock:
            path = self._safe_dir(thread_id, collection_id)
            path.mkdir(parents=True, exist_ok=True)
            faiss.write_index(index, str(path / "index.faiss"))
            (path / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
            manifest = {
                "format_version": FORMAT_VERSION,
                "thread_id": thread_id,
                "document_id": collection_id,
                **fields,
                "chunks": len(chunks),
                "embedding_model": self.embedding_model,
                "dimensions": int(matrix.shape[1]),
                "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "public_fields": sorted(fields),
                "sha256": {
                    "index.faiss": _sha256(path / "index.faiss"),
                    "chunks.json": _sha256(path / "chunks.json"),
                },
            }
            (path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            self._cache[(thread_id, collection_id)] = _LoadedIndex(index, chunks, manifest)
        return self._public(manifest)

    def collection_exists(self, thread_id: str, collection_id: str) -> bool:
        return self._read_manifest(thread_id, collection_id) is not None

    # ---------------- read ----------------
    @staticmethod
    def _public(manifest: dict) -> dict:
        keys = ("document_id", "filename", "pages", "chunks", "created_at", "embedding_model", "file_sha256",
                *manifest.get("public_fields", []))
        return {k: manifest.get(k) for k in keys if k in manifest} | {"status": "indexed"}

    def _read_manifest(self, thread_id: str, document_id: str) -> dict | None:
        path = self._safe_dir(thread_id, document_id) / "manifest.json"
        if not path.exists():
            return None
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if manifest.get("thread_id") != thread_id or manifest.get("document_id") != document_id:
            log_event("index_ownership_mismatch", level=30, document_id=document_id)
            return None
        return manifest

    def list_documents(self, thread_id: str) -> list[dict]:
        try:
            thread_dir = self._safe_dir(thread_id)
        except DocumentError:
            return []
        if not thread_dir.is_dir():
            return []
        docs = []
        for child in sorted(thread_dir.iterdir()):
            if child.is_dir() and _ID_RE.match(child.name):
                manifest = self._read_manifest(thread_id, child.name)
                if manifest and manifest.get("format_version") == FORMAT_VERSION:
                    docs.append(self._public(manifest))
        return sorted(docs, key=lambda d: d.get("created_at") or "")

    def find_by_hash(self, thread_id: str, file_sha256: str) -> dict | None:
        return next((d for d in self.list_documents(thread_id) if d.get("file_sha256") == file_sha256), None)

    def _load(self, thread_id: str, document_id: str) -> _LoadedIndex:
        key = (thread_id, document_id)
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            manifest = self._read_manifest(thread_id, document_id)
            if manifest is None:
                raise IndexIntegrityError(f"missing/foreign manifest for {document_id}")
            if manifest.get("embedding_model") != self.embedding_model:
                raise IndexIntegrityError(
                    f"index built with {manifest.get('embedding_model')}, current {self.embedding_model}",
                    user_message="This document was indexed with a different embedding model; please re-upload it.",
                )
            path = self._safe_dir(thread_id, document_id)
            for fname, expected in (manifest.get("sha256") or {}).items():
                if fname not in ("index.faiss", "chunks.json") or _sha256(path / fname) != expected:
                    raise IndexIntegrityError(f"checksum mismatch for {fname} in {document_id}")
            index = faiss.read_index(str(path / "index.faiss"))
            chunks = json.loads((path / "chunks.json").read_text(encoding="utf-8"))
            if index.ntotal != len(chunks):
                raise IndexIntegrityError(f"index/chunk count mismatch in {document_id}")
            loaded = _LoadedIndex(index, chunks, manifest)
            self._cache[key] = loaded
            return loaded

    def search(
        self,
        thread_id: str,
        query_vector: list[float],
        k: int,
        document_ids: list[str] | None = None,
    ) -> list[tuple[dict[str, Any], float]]:
        """Cosine top-k across this thread's documents (optionally filtered)."""
        q = _normalise(np.asarray([query_vector], dtype="float32"))
        hits: list[tuple[dict[str, Any], float]] = []
        for doc in self.list_documents(thread_id):
            if document_ids and doc["document_id"] not in document_ids:
                continue
            try:
                loaded = self._load(thread_id, doc["document_id"])
            except IndexIntegrityError as exc:
                log_event("index_rejected", level=40, document_id=doc["document_id"], detail=exc.detail)
                continue
            if loaded.index.d != q.shape[1]:
                continue
            scores, ids = loaded.index.search(q, min(k, loaded.index.ntotal))
            hits.extend(
                (loaded.chunks[i], float(s)) for s, i in zip(scores[0], ids[0]) if i >= 0
            )
        hits.sort(key=lambda h: h[1], reverse=True)
        return hits[:k]

    # ---------------- delete ----------------
    def delete_thread(self, thread_id: str) -> None:
        with self._lock:
            for key in [k for k in self._cache if k[0] == thread_id]:
                self._cache.pop(key, None)
            try:
                shutil.rmtree(self._safe_dir(thread_id), ignore_errors=True)
            except DocumentError:
                pass

    def delete_collection(self, thread_id: str, document_id: str) -> bool:
        return self.delete_document(thread_id, document_id)

    def delete_document(self, thread_id: str, document_id: str) -> bool:
        with self._lock:
            self._cache.pop((thread_id, document_id), None)
            path = self._safe_dir(thread_id, document_id)
            existed = path.exists()
            shutil.rmtree(path, ignore_errors=True)
            return existed
