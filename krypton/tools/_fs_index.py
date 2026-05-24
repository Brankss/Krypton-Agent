"""Persistent FTS5 file index + Everything (voidtools) detection.

Two-tier instant-find backend used by InstantFindTool:

  1. Everything (es.exe) — voidtools' NTFS USN-journal indexer. If
     installed, queries are answered in < 50 ms straight from RAM, with
     whole-drive coverage. Best path on Windows.

  2. SQLite FTS5 index — self-built, lazy. First scan of a root costs
     O(n) syscalls (10-60s for a typical Krypton workdir) but every
     subsequent query is < 5 ms. Cross-platform.

If neither is available, callers fall back to the scandir-based
FindFilesTool.
"""
from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


# Same prune list as filesystem.py — keep them in sync.
_PRUNE_DIRS = frozenset({
    ".git", ".hg", ".svn",
    "node_modules", "bower_components",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".venv", "venv", "env", ".tox",
    "dist", "build", ".next", ".nuxt", ".cache",
    ".idea", ".vscode",
    "target", "Pods",
})


_SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    id      INTEGER PRIMARY KEY,
    path    TEXT NOT NULL UNIQUE,
    name    TEXT NOT NULL,
    parent  TEXT NOT NULL,
    is_dir  INTEGER NOT NULL,
    mtime   INTEGER NOT NULL,
    size    INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_files_name ON files(name COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_files_parent ON files(parent);

CREATE VIRTUAL TABLE IF NOT EXISTS files_fts USING fts5(
    path, name,
    content='files',
    content_rowid='id',
    tokenize='unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS files_ai AFTER INSERT ON files BEGIN
    INSERT INTO files_fts(rowid, path, name) VALUES (new.id, new.path, new.name);
END;

CREATE TRIGGER IF NOT EXISTS files_ad AFTER DELETE ON files BEGIN
    INSERT INTO files_fts(files_fts, rowid, path, name) VALUES('delete', old.id, old.path, old.name);
END;

CREATE TRIGGER IF NOT EXISTS files_au AFTER UPDATE ON files BEGIN
    INSERT INTO files_fts(files_fts, rowid, path, name) VALUES('delete', old.id, old.path, old.name);
    INSERT INTO files_fts(rowid, path, name) VALUES (new.id, new.path, new.name);
END;

CREATE TABLE IF NOT EXISTS roots (
    root        TEXT PRIMARY KEY,
    last_scan   INTEGER NOT NULL,
    file_count  INTEGER NOT NULL
);
"""


# ===========================================================================
# Everything (voidtools) — preferred backend on Windows
# ===========================================================================


def detect_everything() -> str | None:
    """Return absolute path to es.exe if installed, else None.

    Looks in PATH first, then the standard install locations.
    """
    if hit := shutil.which("es"):
        return hit
    candidates = [
        r"C:\Program Files\Everything\es.exe",
        r"C:\Program Files (x86)\Everything\es.exe",
        r"C:\Tools\es\es.exe",
        r"C:\Tools\Everything\es.exe",
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    return None


def everything_search(
    es_exe: str,
    query: str,
    *,
    root: str | None = None,
    limit: int = 200,
    kind: str | None = None,
) -> list[str]:
    """Query Everything via es.exe. Returns absolute paths sorted by mtime DESC."""
    cmd = [es_exe, "-n", str(limit), "-sort", "date-modified-descending"]
    if kind == "file":
        cmd.append("/a-d")
    elif kind == "dir":
        cmd.append("/ad")

    # Compose query: optional path: prefix to scope, then the user pattern
    pieces: list[str] = []
    if root:
        pieces.append(f'path:"{os.path.abspath(root)}"')
    pieces.append(query)
    cmd.append(" ".join(pieces))

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
            encoding="utf-8",
            errors="ignore",
        )
    except (subprocess.TimeoutExpired, OSError):
        return []
    if proc.returncode != 0:
        return []
    return [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()][:limit]


# ===========================================================================
# SQLite FTS5 index
# ===========================================================================


class FileIndex:
    """Persistent path index backed by SQLite + FTS5."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        c = sqlite3.connect(self.db_path, timeout=30.0)
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=NORMAL")
        c.execute("PRAGMA temp_store=MEMORY")
        try:
            yield c
            c.commit()
        finally:
            c.close()

    def _init_schema(self) -> None:
        with self._conn() as c:
            c.executescript(_SCHEMA)

    # ------------------------------------------------------------------
    def root_age(self, root: str) -> float | None:
        """Seconds since this root was last fully scanned, or None if never."""
        root = os.path.abspath(root)
        with self._conn() as c:
            r = c.execute(
                "SELECT last_scan FROM roots WHERE root = ?", (root,)
            ).fetchone()
        return None if not r else time.time() - r[0]

    def file_count(self, root: str) -> int:
        root = os.path.abspath(root)
        with self._conn() as c:
            r = c.execute(
                "SELECT file_count FROM roots WHERE root = ?", (root,)
            ).fetchone()
        return r[0] if r else 0

    # ------------------------------------------------------------------
    def build(self, root: str, *, include_hidden: bool = False) -> tuple[int, float]:
        """Scan root recursively, populate index. Returns (file_count, seconds)."""
        root = os.path.abspath(root)
        t0 = time.perf_counter()
        count = 0
        batch: list[tuple] = []
        with self._conn() as c:
            # Wipe any prior entries strictly under this root
            c.execute(
                "DELETE FROM files WHERE path = ? OR path LIKE ?",
                (root, root.rstrip(os.sep) + os.sep + "%"),
            )
            for row in _walk(root, include_hidden=include_hidden):
                batch.append(row)
                if len(batch) >= 5000:
                    self._flush(c, batch)
                    count += len(batch)
                    batch.clear()
            if batch:
                self._flush(c, batch)
                count += len(batch)
            c.execute(
                "INSERT OR REPLACE INTO roots (root, last_scan, file_count) VALUES (?, ?, ?)",
                (root, int(time.time()), count),
            )
        return count, time.perf_counter() - t0

    @staticmethod
    def _flush(c: sqlite3.Connection, batch: list[tuple]) -> None:
        c.executemany(
            "INSERT OR REPLACE INTO files "
            "(path, name, parent, is_dir, mtime, size) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            batch,
        )

    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        *,
        root: str | None = None,
        limit: int = 100,
        kind: str | None = None,
    ) -> list[tuple[str, int, int]]:
        """Returns [(path, mtime, is_dir), ...] sorted by mtime DESC."""
        wheres: list[str] = []
        params: list = []

        # Glob → SQL LIKE on basename. Plain text → FTS5 prefix match.
        is_glob = any(ch in query for ch in "*?[")
        if is_glob:
            like = (
                query.replace("\\", "\\\\")
                .replace("%", "\\%")
                .replace("_", "\\_")
                .replace("*", "%")
                .replace("?", "_")
            )
            wheres.append("name LIKE ? ESCAPE '\\' COLLATE NOCASE")
            params.append(like)
        else:
            # FTS5 prefix search on path+name. Quote to escape special chars.
            fts_q = '"' + query.replace('"', '""') + '"*'
            wheres.append(
                "id IN (SELECT rowid FROM files_fts WHERE files_fts MATCH ?)"
            )
            params.append(fts_q)

        if root:
            root = os.path.abspath(root)
            wheres.append("(path = ? OR path LIKE ?)")
            params.extend([root, root.rstrip(os.sep) + os.sep + "%"])

        if kind == "file":
            wheres.append("is_dir = 0")
        elif kind == "dir":
            wheres.append("is_dir = 1")

        sql = "SELECT path, mtime, is_dir FROM files"
        if wheres:
            sql += " WHERE " + " AND ".join(wheres)
        sql += " ORDER BY mtime DESC LIMIT ?"
        params.append(limit)

        with self._conn() as c:
            return c.execute(sql, params).fetchall()


def _walk(root: str, *, include_hidden: bool) -> Iterator[tuple]:
    """Yield (path, name, parent, is_dir, mtime, size) for everything under root."""
    stack: list[str] = [root]
    while stack:
        cur = stack.pop()
        try:
            it = os.scandir(cur)
        except (PermissionError, FileNotFoundError, OSError):
            continue
        with it:
            for entry in it:
                name = entry.name
                if not include_hidden and name.startswith("."):
                    continue
                try:
                    is_dir = entry.is_dir(follow_symlinks=False)
                    st = entry.stat(follow_symlinks=False)
                except OSError:
                    continue
                if is_dir:
                    if name in _PRUNE_DIRS:
                        continue
                    stack.append(entry.path)
                yield (
                    entry.path,
                    name,
                    cur,
                    1 if is_dir else 0,
                    int(st.st_mtime),
                    st.st_size,
                )
