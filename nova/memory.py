"""
Persistence: LangGraph SQLite checkpointer + a small thread registry.

* Conversation state lives in LangGraph checkpoints (`SqliteSaver`), keyed by thread_id,
  which gives restart persistence and thread isolation for free.
* `threads` is an indexed registry table (title, timestamps, message count) so the
  sidebar/API can list conversations without scanning every checkpoint.
* SQLite runs in WAL mode with a busy timeout; SqliteSaver serialises writes with its
  own lock, the registry with ours. This is appropriate for a single-node deployment —
  see ADR-003 for when to move to PostgreSQL (`langgraph-checkpoint-postgres`).
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def make_checkpointer(conn: sqlite3.Connection) -> SqliteSaver:
    saver = SqliteSaver(conn)
    saver.setup()
    return saver


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class ThreadRegistry:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._lock = threading.Lock()
        with self._lock, self._conn:
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS threads (
                       thread_id   TEXT PRIMARY KEY,
                       title       TEXT NOT NULL,
                       created_at  TEXT NOT NULL,
                       updated_at  TEXT NOT NULL,
                       turns       INTEGER NOT NULL DEFAULT 0
                   )"""
            )
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_threads_updated ON threads(updated_at DESC)")

    def touch(self, thread_id: str, message: str, *, count_turn: bool = True) -> None:
        """Register activity. The first user message becomes the title (a placeholder
        title from a document upload is replaced when the first message arrives)."""
        title = " ".join(message.split())[:80] or "New conversation"
        now = _now()
        inc = 1 if count_turn else 0
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT INTO threads(thread_id, title, created_at, updated_at, turns)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(thread_id) DO UPDATE SET
                       updated_at = excluded.updated_at,
                       title = CASE WHEN threads.turns = 0 AND ? = 1 THEN excluded.title ELSE threads.title END,
                       turns = threads.turns + ?""",
                (thread_id, title, now, now, inc, inc, inc),
            )

    def list(self, limit: int = 100) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT thread_id, title, created_at, updated_at, turns FROM threads ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(zip(("thread_id", "title", "created_at", "updated_at", "turns"), r)) for r in rows]

    def get(self, thread_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT thread_id, title, created_at, updated_at, turns FROM threads WHERE thread_id=?",
                (thread_id,),
            ).fetchone()
        return dict(zip(("thread_id", "title", "created_at", "updated_at", "turns"), row)) if row else None

    def delete(self, thread_id: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM threads WHERE thread_id=?", (thread_id,))

    def is_empty(self) -> bool:
        with self._lock:
            return self._conn.execute("SELECT 1 FROM threads LIMIT 1").fetchone() is None
