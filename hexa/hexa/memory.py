"""SQLite-based memory for Hexa (notes and facts)."""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    tags TEXT DEFAULT '[]',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_memories_content ON memories(content);
"""


class Memory:
    """Simple SQLite-backed memory store."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            conn.commit()

    @contextmanager
    def _connect(self):
        """Get a database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def remember(self, text: str, tags: list[str] | None = None) -> int:
        """Store a note/fact in memory."""
        now = time.time()
        tags_json = json.dumps(tags or [])
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO memories (content, tags, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (text, tags_json, now, now)
            )
            conn.commit()
            return cursor.lastrowid

    def recall(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search memories by content (simple LIKE search)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, content, tags, created_at, updated_at FROM memories "
                "WHERE content LIKE ? ORDER BY created_at DESC LIMIT ?",
                (f"%{query}%", limit)
            ).fetchall()
            return [
                {
                    "id": row["id"],
                    "content": row["content"],
                    "tags": json.loads(row["tags"]),
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
                for row in rows
            ]

    def get_all(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get all memories, most recent first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, content, tags, created_at, updated_at FROM memories "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
            return [
                {
                    "id": row["id"],
                    "content": row["content"],
                    "tags": json.loads(row["tags"]),
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
                for row in rows
            ]

    def delete(self, memory_id: int) -> bool:
        """Delete a memory by ID."""
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            conn.commit()
            return cursor.rowcount > 0


# Global memory instance
_memory: Memory | None = None


def get_memory(db_path: str | Path | None = None) -> Memory:
    """Get the global memory instance."""
    global _memory
    if _memory is None:
        if db_path is None:
            from hexa.config import get_config
            db_path = get_config().memory.db_path
        _memory = Memory(db_path)
    return _memory