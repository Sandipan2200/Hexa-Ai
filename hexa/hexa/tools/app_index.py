"""Persistent application index for Hexa.

Stores discovered applications in SQLite (reusing hexa.db) with support for:
- Stable application ID
- Display name, normalized name
- Publisher, version
- Executable path or UWP AUMID
- Discovery source
- Search aliases
- Last discovered timestamp

Also provides intelligent matching: exact, normalized, alias, fuzzy.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path
from typing import Any

from hexa.tools.app_discovery import DiscoveredApp

logger = logging.getLogger(__name__)

APP_INDEX_SCHEMA = """
CREATE TABLE IF NOT EXISTS app_index (
    app_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    publisher TEXT,
    version TEXT,
    executable TEXT,
    aumid TEXT,
    source TEXT NOT NULL,
    aliases TEXT DEFAULT '[]',
    last_discovered REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_app_index_normalized ON app_index(normalized_name);
CREATE INDEX IF NOT EXISTS idx_app_index_source ON app_index(source);
"""

PREFERRED_APPS_SCHEMA = """
CREATE TABLE IF NOT EXISTS preferred_apps (
    category TEXT PRIMARY KEY,  -- e.g., "browser", "editor", "terminal", "music"
    app_id TEXT NOT NULL,
    updated_at REAL NOT NULL
);
"""


@dataclass(eq=True, frozen=True)
class IndexedApp:
    """Application record from the persistent index.

    Frozen and hashable by app_id for use in sets.
    """
    app_id: str
    display_name: str
    normalized_name: str
    publisher: str | None
    version: str | None
    executable: str | None
    aumid: str | None
    source: str
    aliases: list[str]
    last_discovered: float

    def __hash__(self) -> int:
        return hash(self.app_id)

    def to_discovered_app(self) -> DiscoveredApp:
        """Convert to DiscoveredApp for compatibility."""
        return DiscoveredApp(
            app_id=self.app_id,
            display_name=self.display_name,
            normalized_name=self.normalized_name,
            publisher=self.publisher,
            version=self.version,
            executable=self.executable,
            aumid=self.aumid,
            source=self.source,
            aliases=self.aliases,
            last_discovered=self.last_discovered,
        )

    def launch_command(self) -> list[str] | None:
        """Return a safe launch command (list, no shell=True)."""
        if self.aumid:
            # UWP app: launch via explorer.exe shell:AppsFolder\AUMID
            return ["explorer.exe", f"shell:AppsFolder\\{self.aumid}"]
        if self.executable:
            return [self.executable]
        return None


class AppIndex:
    """Persistent SQLite-backed application index with intelligent matching."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        with self._connect() as conn:
            conn.executescript(APP_INDEX_SCHEMA)
            conn.executescript(PREFERRED_APPS_SCHEMA)
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

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------
    def upsert_app(self, app: DiscoveredApp) -> None:
        """Insert or update an application record."""
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO app_index (app_id, display_name, normalized_name, publisher,
                                       version, executable, aumid, source, aliases, last_discovered)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(app_id) DO UPDATE SET
                    display_name=excluded.display_name,
                    normalized_name=excluded.normalized_name,
                    publisher=excluded.publisher,
                    version=excluded.version,
                    executable=excluded.executable,
                    aumid=excluded.aumid,
                    source=excluded.source,
                    aliases=excluded.aliases,
                    last_discovered=excluded.last_discovered
                """,
                (
                    app.app_id,
                    app.display_name,
                    app.normalized_name,
                    app.publisher,
                    app.version,
                    app.executable,
                    app.aumid,
                    app.source,
                    json.dumps(app.aliases),
                    now,
                ),
            )
            conn.commit()

    def bulk_upsert(self, apps: dict[str, DiscoveredApp]) -> int:
        """Bulk insert/update multiple apps. Returns count of affected rows."""
        count = 0
        with self._connect() as conn:
            for app in apps.values():
                cursor = conn.execute(
                    """
                    INSERT INTO app_index (app_id, display_name, normalized_name, publisher,
                                           version, executable, aumid, source, aliases, last_discovered)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(app_id) DO UPDATE SET
                        display_name=excluded.display_name,
                        normalized_name=excluded.normalized_name,
                        publisher=excluded.publisher,
                        version=excluded.version,
                        executable=excluded.executable,
                        aumid=excluded.aumid,
                        source=excluded.source,
                        aliases=excluded.aliases,
                        last_discovered=excluded.last_discovered
                    """,
                    (
                        app.app_id,
                        app.display_name,
                        app.normalized_name,
                        app.publisher,
                        app.version,
                        app.executable,
                        app.aumid,
                        app.source,
                        json.dumps(app.aliases),
                        app.last_discovered,
                    ),
                )
                count += cursor.rowcount
            conn.commit()
        return count

    def delete_app(self, app_id: str) -> bool:
        """Delete an application from the index."""
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM app_index WHERE app_id = ?", (app_id,))
            conn.commit()
            return cursor.rowcount > 0

    def clear(self) -> int:
        """Clear all apps from the index. Returns count deleted."""
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM app_index")
            conn.commit()
            return cursor.rowcount

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------
    def get_app(self, app_id: str) -> IndexedApp | None:
        """Get an app by its stable ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM app_index WHERE app_id = ?", (app_id,)
            ).fetchone()
        return self._row_to_app(row) if row else None

    def get_all(self, limit: int = 500) -> list[IndexedApp]:
        """Get all indexed apps, most recently discovered first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM app_index ORDER BY last_discovered DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_app(r) for r in rows]

    def get_by_source(self, source: str) -> list[IndexedApp]:
        """Get apps from a specific discovery source."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM app_index WHERE source = ? ORDER BY display_name", (source,)
            ).fetchall()
        return [self._row_to_app(r) for r in rows]

    def count(self) -> int:
        """Return total number of indexed apps."""
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM app_index").fetchone()
        return row[0] if row else 0

    # ------------------------------------------------------------------
    # Intelligent matching
    # ------------------------------------------------------------------
    def find_best_match(self, query: str, threshold: float = 0.6) -> list[IndexedApp]:
        """Find best matching apps for a query.

        Priority:
        1. Exact app_id match
        2. Exact display_name match (case-insensitive)
        3. Exact normalized_name match
        4. Alias match
        5. Prefix match on normalized name
        6. Fuzzy match (difflib) above threshold

        Returns list of matches (empty if none), best first.
        """
        query_lower = query.lower().strip()
        query_norm = self._normalize(query_lower)

        all_apps = self.get_all()

        # 1. Exact app_id
        for app in all_apps:
            if app.app_id.lower() == query_lower:
                return [app]

        # 2. Exact display_name
        for app in all_apps:
            if app.display_name.lower() == query_lower:
                return [app]

        # 3. Exact normalized_name
        for app in all_apps:
            if app.normalized_name == query_norm:
                return [app]

        # 4. Alias match
        for app in all_apps:
            if any(query_lower == alias.lower() for alias in app.aliases):
                return [app]

        # 5. Prefix match on normalized name
        prefix_matches = [app for app in all_apps if app.normalized_name.startswith(query_norm)]
        if prefix_matches:
            return prefix_matches

        # 6. Fuzzy match on display_name and normalized_name
        names = [app.display_name for app in all_apps]
        close = get_close_matches(query_lower, names, n=5, cutoff=threshold)
        if close:
            # Return in order of closeness
            matched = []
            for name in close:
                for app in all_apps:
                    if app.display_name == name:
                        matched.append(app)
                        break
            return matched

        # Also try fuzzy on normalized names
        norm_names = [app.normalized_name for app in all_apps]
        close_norm = get_close_matches(query_norm, norm_names, n=5, cutoff=threshold)
        if close_norm:
            matched = []
            for name in close_norm:
                for app in all_apps:
                    if app.normalized_name == name:
                        matched.append(app)
                        break
            return matched

        return []

    def resolve_ambiguous(self, query: str, max_results: int = 3) -> list[IndexedApp]:
        """Find multiple close matches for disambiguation."""
        query_lower = query.lower().strip()
        query_norm = self._normalize(query_lower)

        all_apps = self.get_all()

        # Collect candidates: prefix + fuzzy
        candidates = set()

        # Prefix matches
        for app in all_apps:
            if app.normalized_name.startswith(query_norm) or app.display_name.lower().startswith(query_lower):
                candidates.add(app)

        # Fuzzy matches
        names = [app.display_name for app in all_apps]
        for name in get_close_matches(query_lower, names, n=max_results, cutoff=0.5):
            for app in all_apps:
                if app.display_name == name:
                    candidates.add(app)

        norm_names = [app.normalized_name for app in all_apps]
        for name in get_close_matches(query_norm, norm_names, n=max_results, cutoff=0.5):
            for app in all_apps:
                if app.normalized_name == name:
                    candidates.add(app)

        # Sort by relevance (shorter normalized name first, then by last_discovered)
        sorted_candidates = sorted(
            candidates,
            key=lambda a: (len(a.normalized_name), -a.last_discovered)
        )
        return sorted_candidates[:max_results]

    def get_preferred(self, category: str) -> IndexedApp | None:
        """Get the preferred app for a generic category (e.g., 'browser')."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT app_id FROM preferred_apps WHERE category = ?", (category,)
            ).fetchone()
        if row:
            return self.get_app(row["app_id"])
        return None

    def set_preferred(self, category: str, app_id: str) -> bool:
        """Set the preferred app for a category."""
        app = self.get_app(app_id)
        if not app:
            return False
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO preferred_apps (category, app_id, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(category) DO UPDATE SET
                    app_id=excluded.app_id, updated_at=excluded.updated_at
                """,
                (category, app_id, time.time()),
            )
            conn.commit()
        return True

    def get_all_preferred(self) -> dict[str, IndexedApp]:
        """Get all category -> preferred app mappings."""
        with self._connect() as conn:
            rows = conn.execute("SELECT category, app_id FROM preferred_apps").fetchall()
        result = {}
        for row in rows:
            app = self.get_app(row["app_id"])
            if app:
                result[row["category"]] = app
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _row_to_app(self, row: sqlite3.Row) -> IndexedApp:
        return IndexedApp(
            app_id=row["app_id"],
            display_name=row["display_name"],
            normalized_name=row["normalized_name"],
            publisher=row["publisher"],
            version=row["version"],
            executable=row["executable"],
            aumid=row["aumid"],
            source=row["source"],
            aliases=json.loads(row["aliases"] or "[]"),
            last_discovered=row["last_discovered"],
        )

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize for matching: lowercase, alphanumeric only."""
        import re
        return re.sub(r"[^a-z0-9\s]", "", text.lower()).strip()


# Global instance
_index: AppIndex | None = None
_index_db_path: Path | None = None


def get_app_index(db_path: str | Path | None = None) -> AppIndex:
    """Get the global AppIndex instance.

    If db_path differs from the cached instance, a new instance is created.
    """
    global _index, _index_db_path
    if db_path is None:
        from hexa.config import get_config
        db_path = get_config().memory.db_path

    db_path = Path(db_path)
    if _index is None or _index_db_path != db_path:
        _index = AppIndex(db_path)
        _index_db_path = db_path
    return _index