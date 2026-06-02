"""SQLite-backed memory for Krypton.

Two tables:
  * patterns   — reusable knowledge the agent learns (best-practice recipes,
                 known errors + fixes, optimizations). Keyword-retrievable.
  * facts      — durable user/project facts surfaced into the system prompt.

The whole API is sync because SQLite is fast for our scale, but every
call is wrapped through asyncio.to_thread to avoid blocking the loop.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

PatternKind = Literal["recipe", "known_error", "optimization", "preference"]


@dataclass(slots=True)
class Pattern:
    id: int
    kind: PatternKind
    title: str
    body: str
    tags: list[str] = field(default_factory=list)
    uses: int = 0
    created_at: float = 0.0
    updated_at: float = 0.0

    def render(self) -> str:
        tagstr = (" #" + " #".join(self.tags)) if self.tags else ""
        return f"[{self.kind}{tagstr}] {self.title}\n  {self.body}"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS patterns (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT NOT NULL,
    title       TEXT NOT NULL,
    body        TEXT NOT NULL,
    tags_json   TEXT NOT NULL DEFAULT '[]',
    uses        INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_patterns_kind ON patterns(kind);
CREATE INDEX IF NOT EXISTS idx_patterns_updated ON patterns(updated_at DESC);

CREATE TABLE IF NOT EXISTS facts (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    updated_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS conversations (
    chat_id     INTEGER PRIMARY KEY,
    payload     TEXT NOT NULL,
    updated_at  REAL NOT NULL
);
"""


SCHEMA_VERSION = 2  # bump when _SCHEMA changes and add a migration step in _migrate


class MemoryStore:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._migrate()
        self._lock = asyncio.Lock()

    def _migrate(self) -> None:
        cur = self._conn.execute("PRAGMA user_version;")
        current = cur.fetchone()[0] or 0
        if current >= SCHEMA_VERSION:
            return
        # v0 -> v1: add conversations table (already in _SCHEMA, idempotent)
        # v1 -> v2: no structural change, just a version marker for future
        # Future migrations go here, each guarded by `if current < N`.
        self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION};")

    # ----- patterns -----------------------------------------------------
    async def add_pattern(
        self,
        kind: PatternKind,
        title: str,
        body: str,
        tags: Iterable[str] = (),
    ) -> Pattern:
        now = time.time()
        tags_l = sorted({t.lower().strip() for t in tags if t.strip()})
        async with self._lock:
            cur = await asyncio.to_thread(
                self._conn.execute,
                "INSERT INTO patterns(kind,title,body,tags_json,created_at,updated_at) VALUES(?,?,?,?,?,?)",
                (kind, title, body, json.dumps(tags_l), now, now),
            )
            pid = cur.lastrowid or 0
        return Pattern(id=pid, kind=kind, title=title, body=body, tags=tags_l,
                       created_at=now, updated_at=now)

    async def update_pattern(self, pid: int, *, body: str | None = None,
                              tags: Iterable[str] | None = None) -> bool:
        sets, vals = [], []
        if body is not None:
            sets.append("body=?")
            vals.append(body)
        if tags is not None:
            sets.append("tags_json=?")
            vals.append(json.dumps(sorted({t.lower().strip() for t in tags if t.strip()})))
        if not sets:
            return False
        sets.append("updated_at=?")
        vals.append(time.time())
        vals.append(pid)
        async with self._lock:
            cur = await asyncio.to_thread(
                self._conn.execute,
                f"UPDATE patterns SET {','.join(sets)} WHERE id=?",
                tuple(vals),
            )
            return cur.rowcount > 0

    async def delete_pattern(self, pid: int) -> bool:
        async with self._lock:
            cur = await asyncio.to_thread(self._conn.execute, "DELETE FROM patterns WHERE id=?", (pid,))
            return cur.rowcount > 0

    async def touch_pattern(self, pid: int) -> None:
        async with self._lock:
            await asyncio.to_thread(
                self._conn.execute,
                "UPDATE patterns SET uses=uses+1, updated_at=? WHERE id=?",
                (time.time(), pid),
            )

    async def list_patterns(self, kind: PatternKind | None = None, limit: int = 100) -> list[Pattern]:
        q = "SELECT * FROM patterns"
        params: tuple = ()
        if kind:
            q += " WHERE kind=?"
            params = (kind,)
        q += " ORDER BY updated_at DESC LIMIT ?"
        params = (*params, limit)
        rows = await asyncio.to_thread(lambda: self._conn.execute(q, params).fetchall())
        return [_row_to_pattern(r) for r in rows]

    async def search_patterns(self, query: str, limit: int = 10) -> list[Pattern]:
        """Two-stage retrieval:
          1. SQL LIKE union over title/body/tags for each token — pulls only
             rows that contain at least one token (small candidate set).
          2. Python rerank: count distinct token hits + small `uses` boost.
        Scales much better than loading 500 rows every search.
        """
        tokens = [t for t in _tokenize(query) if len(t) >= 3][:8]
        if not tokens:
            return []

        clauses = []
        params: list = []
        for t in tokens:
            like = f"%{t}%"
            clauses.append("LOWER(title) LIKE ? OR LOWER(body) LIKE ? OR LOWER(tags_json) LIKE ?")
            params.extend([like, like, like])
        where = " OR ".join(f"({c})" for c in clauses)
        sql = f"SELECT * FROM patterns WHERE {where} ORDER BY updated_at DESC LIMIT 100"

        rows = await asyncio.to_thread(lambda: self._conn.execute(sql, params).fetchall())
        scored: list[tuple[float, Pattern]] = []
        for r in rows:
            p = _row_to_pattern(r)
            hay = (p.title + " " + p.body + " " + " ".join(p.tags)).lower()
            hits = sum(1 for t in tokens if t in hay)
            if hits == 0:
                continue
            score = hits + min(p.uses, 10) * 0.1
            scored.append((score, p))
        scored.sort(key=lambda s: s[0], reverse=True)
        return [p for _, p in scored[:limit]]

    async def remember(
        self,
        kind: PatternKind,
        title: str,
        body: str,
        tags: Iterable[str] = (),
        *,
        sim_threshold: float = 0.55,
    ) -> tuple[Pattern, bool]:
        """Atomically dedup-or-insert a pattern.

        Looks for a near-duplicate (same kind, Jaccard ≥ `sim_threshold` on
        title tokens) and merges into it; otherwise inserts a new row. The
        whole check-then-act runs under the write lock in one worker thread,
        so concurrent calls (parallel tool dispatch) can NEVER both pass the
        duplicate check and double-insert — closing the TOCTOU that the old
        search-then-add path had.

        Returns `(pattern, merged)` — `merged` is True when an existing
        pattern was updated rather than a new one created.
        """
        tags_l = sorted({t.lower().strip() for t in tags if t.strip()})
        async with self._lock:
            return await asyncio.to_thread(
                self._remember_sync, kind, title.strip(), body.strip(), tags_l, sim_threshold
            )

    def _remember_sync(
        self,
        kind: PatternKind,
        title: str,
        body: str,
        tags_l: list[str],
        sim_threshold: float,
    ) -> tuple[Pattern, bool]:
        now = time.time()
        # Candidate search (same shape as search_patterns stage 1) — inline so
        # it shares the locked thread with the write that follows it.
        tokens = [t for t in _tokenize(title + " " + body) if len(t) >= 3][:8]
        candidates: list[Pattern] = []
        if tokens:
            clauses, params = [], []
            for t in tokens:
                like = f"%{t}%"
                clauses.append("LOWER(title) LIKE ? OR LOWER(body) LIKE ? OR LOWER(tags_json) LIKE ?")
                params.extend([like, like, like])
            where = " OR ".join(f"({c})" for c in clauses)
            sql = f"SELECT * FROM patterns WHERE {where} ORDER BY updated_at DESC LIMIT 50"
            candidates = [_row_to_pattern(r) for r in self._conn.execute(sql, params).fetchall()]

        title_tokens = _tokset(title)
        for ex in candidates:
            if ex.kind != kind:
                continue
            if _jaccard(title_tokens, _tokset(ex.title)) >= sim_threshold:
                merged_tags = sorted(set(ex.tags) | set(tags_l))
                new_body = ex.body if body in ex.body else (ex.body.rstrip() + "\n---\n" + body)[:2000]
                self._conn.execute(
                    "UPDATE patterns SET body=?, tags_json=?, updated_at=? WHERE id=?",
                    (new_body, json.dumps(merged_tags), now, ex.id),
                )
                return (
                    Pattern(id=ex.id, kind=ex.kind, title=ex.title, body=new_body,
                            tags=merged_tags, uses=ex.uses, created_at=ex.created_at, updated_at=now),
                    True,
                )

        cur = self._conn.execute(
            "INSERT INTO patterns(kind,title,body,tags_json,created_at,updated_at) VALUES(?,?,?,?,?,?)",
            (kind, title, body, json.dumps(tags_l), now, now),
        )
        pid = cur.lastrowid or 0
        return (
            Pattern(id=pid, kind=kind, title=title, body=body, tags=tags_l,
                    created_at=now, updated_at=now),
            False,
        )

    # ----- facts (key/value) -------------------------------------------
    async def set_fact(self, key: str, value: str) -> None:
        async with self._lock:
            await asyncio.to_thread(
                self._conn.execute,
                "INSERT INTO facts(key,value,updated_at) VALUES(?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (key, value, time.time()),
            )

    async def get_fact(self, key: str) -> str | None:
        row = await asyncio.to_thread(
            lambda: self._conn.execute("SELECT value FROM facts WHERE key=?", (key,)).fetchone()
        )
        return row["value"] if row else None

    async def all_facts(self) -> dict[str, str]:
        rows = await asyncio.to_thread(
            lambda: self._conn.execute("SELECT key,value FROM facts ORDER BY key").fetchall()
        )
        return {r["key"]: r["value"] for r in rows}

    async def delete_fact(self, key: str) -> bool:
        async with self._lock:
            cur = await asyncio.to_thread(self._conn.execute, "DELETE FROM facts WHERE key=?", (key,))
            return cur.rowcount > 0

    # ----- conversations (per-chat persistence) ------------------------
    async def save_conversation(self, chat_id: int, payload: str) -> None:
        async with self._lock:
            await asyncio.to_thread(
                self._conn.execute,
                "INSERT INTO conversations(chat_id,payload,updated_at) VALUES(?,?,?) "
                "ON CONFLICT(chat_id) DO UPDATE SET payload=excluded.payload, updated_at=excluded.updated_at",
                (chat_id, payload, time.time()),
            )

    async def load_conversation(self, chat_id: int) -> str | None:
        row = await asyncio.to_thread(
            lambda: self._conn.execute(
                "SELECT payload FROM conversations WHERE chat_id=?", (chat_id,)
            ).fetchone()
        )
        return row["payload"] if row else None

    async def clear_conversation(self, chat_id: int) -> bool:
        async with self._lock:
            cur = await asyncio.to_thread(
                self._conn.execute, "DELETE FROM conversations WHERE chat_id=?", (chat_id,)
            )
            return cur.rowcount > 0

    async def aclose(self) -> None:
        await asyncio.to_thread(self._conn.close)


def _row_to_pattern(row: sqlite3.Row) -> Pattern:
    try:
        tags = json.loads(row["tags_json"]) or []
    except Exception:  # noqa: BLE001
        tags = []
    return Pattern(
        id=row["id"],
        kind=row["kind"],
        title=row["title"],
        body=row["body"],
        tags=list(tags),
        uses=row["uses"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _tokenize(text: str) -> list[str]:
    import re
    return re.findall(r"[a-z0-9_\-]+", text.lower())


def _tokset(s: str) -> set[str]:
    import re
    return {t for t in re.findall(r"[a-z0-9_]+", s.lower()) if len(t) >= 3}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0
