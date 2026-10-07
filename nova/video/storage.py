"""
Media persistence: an indexed SQLite registry (same database as conversations) plus
filesystem layout and a content-addressed artifact cache.

    data/media/<thread_id>/<media_id>/source.<ext>     uploaded / downloaded source
    data/media_cache/<source_key>/audio.wav            extracted 16 kHz mono audio
                                 transcript-<cfg>.json  timestamped segments
                                 chunks-<cfg>.json + vectors-<cfg>.npy   chunking + embeddings
                                 insights-<cfg>.json    summary / decisions / actions / questions

`source_key` is the SHA-256 of the uploaded bytes (or the YouTube video id), and every
artifact name includes a hash of the configuration that produced it, so re-asking about
the same media never re-transcribes, and changing the Whisper model never reuses a stale
transcript. Cache directories are removed when no media asset references them.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
import threading
from pathlib import Path

from nova.errors import InputValidationError
from nova.video.models import MediaAsset, MediaStatus, utcnow

_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


def config_hash(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16]


class MediaRegistry:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._lock = threading.Lock()
        with self._lock, self._conn:
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS media_assets (
                       media_id   TEXT PRIMARY KEY,
                       thread_id  TEXT NOT NULL,
                       source_key TEXT,
                       status     TEXT NOT NULL,
                       created_at TEXT NOT NULL,
                       data       TEXT NOT NULL
                   )"""
            )
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_media_thread ON media_assets(thread_id)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_media_source ON media_assets(source_key)")

    def save(self, asset: MediaAsset) -> MediaAsset:
        asset.updated_at = utcnow()
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT INTO media_assets(media_id, thread_id, source_key, status, created_at, data)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(media_id) DO UPDATE SET source_key=excluded.source_key,
                       status=excluded.status, data=excluded.data""",
                (asset.media_id, asset.thread_id, asset.source_key, asset.status.value, asset.created_at,
                 asset.model_dump_json()),
            )
        return asset

    def get(self, media_id: str) -> MediaAsset | None:
        with self._lock:
            row = self._conn.execute("SELECT data FROM media_assets WHERE media_id=?", (media_id,)).fetchone()
        return MediaAsset.model_validate_json(row[0]) if row else None

    def list_assets(self, thread_id: str) -> list[MediaAsset]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT data FROM media_assets WHERE thread_id=? ORDER BY created_at", (thread_id,)
            ).fetchall()
        return [MediaAsset.model_validate_json(r[0]) for r in rows]

    def delete(self, media_id: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM media_assets WHERE media_id=?", (media_id,))

    def references(self, source_key: str) -> int:
        with self._lock:
            return self._conn.execute(
                "SELECT COUNT(*) FROM media_assets WHERE source_key=?", (source_key,)
            ).fetchone()[0]

    def interrupted(self) -> list[MediaAsset]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT data FROM media_assets WHERE status=?", (MediaStatus.PROCESSING.value,)
            ).fetchall()
        return [MediaAsset.model_validate_json(r[0]) for r in rows]


class MediaFiles:
    """Path management with the same id validation / containment rules as the vector store."""

    def __init__(self, media_root: Path, cache_root: Path, data_root: Path) -> None:
        self.media_root = media_root.resolve()
        self.cache_root = cache_root.resolve()
        self.data_root = data_root.resolve()
        for p in (self.media_root, self.cache_root):
            p.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _check(*parts: str) -> None:
        for part in parts:
            if not _ID_RE.match(part or ""):
                raise InputValidationError(f"invalid id {part!r}", user_message="Invalid conversation or media id.")

    def _inside(self, root: Path, *parts: str) -> Path:
        self._check(*parts)
        path = root.joinpath(*parts).resolve()
        if root not in path.parents:
            raise InputValidationError("path escapes media root")
        return path

    def asset_dir(self, thread_id: str, media_id: str) -> Path:
        return self._inside(self.media_root, thread_id, media_id)

    def cache_dir(self, source_key: str) -> Path:
        return self._inside(self.cache_root, source_key)

    def relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.data_root).as_posix()

    def absolute(self, relative: str) -> Path:
        path = (self.data_root / relative).resolve()
        if self.data_root not in path.parents:
            raise InputValidationError("path escapes data root")
        return path

    def write_json(self, path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)  # atomic: a crash never leaves a half-written cache entry

    def remove_asset_dir(self, thread_id: str, media_id: str) -> None:
        shutil.rmtree(self.asset_dir(thread_id, media_id), ignore_errors=True)

    def remove_cache(self, source_key: str) -> None:
        shutil.rmtree(self.cache_dir(source_key), ignore_errors=True)
