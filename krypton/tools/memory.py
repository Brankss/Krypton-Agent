"""Tools that let the agent write to / read from its long-term memory.

These are how Krypton learns:
  * remember_pattern — store a reusable recipe / known-error / optimization
  * recall_patterns  — search what it already knows
  * forget_pattern   — drop a stale one
  * note_fact / get_fact — small key/value facts (user prefs, project state)
"""
from __future__ import annotations

from typing import Any

from krypton.memory.store import MemoryStore
from krypton.tools.base import BaseTool, ToolResult


class RememberPatternTool(BaseTool):
    name = "remember_pattern"
    description = (
        "Store a reusable lesson in long-term memory. Use when you've discovered "
        "a non-obvious recipe, hit a known error and want to avoid it next time, "
        "or found an optimization worth keeping. Tag generously for retrieval. "
        "Auto-dedups: if a very similar pattern already exists, it's merged."
    )
    parameters = {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "enum": ["recipe", "known_error", "optimization", "preference"],
            },
            "title": {"type": "string", "description": "Short headline (< 80 chars)."},
            "body": {"type": "string", "description": "Detailed lesson + why + how to apply."},
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "default": [],
                "description": "Keywords for retrieval (e.g. ['ollama', 'http', 'timeout']).",
            },
        },
        "required": ["kind", "title", "body"],
    }
    timeout_s = 5.0

    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    async def run(self, kind: str, title: str, body: str, tags: list[str] | None = None) -> ToolResult:
        # Atomic dedup-or-insert lives in the store: the search + merge/insert
        # runs under one lock so parallel tool dispatch can't double-insert.
        pattern, merged = await self._store.remember(kind, title, body, tags or [])  # type: ignore[arg-type]
        verb = "merged into" if merged else "stored"
        return ToolResult.success(f"{verb} pattern #{pattern.id}: {pattern.title}")


class RecallPatternsTool(BaseTool):
    name = "recall_patterns"
    description = (
        "Search long-term memory for patterns matching a free-text query. "
        "Returns the top matches with titles and bodies."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "default": 6},
        },
        "required": ["query"],
    }
    timeout_s = 5.0

    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    async def run(self, query: str, limit: int = 6) -> ToolResult:
        hits = await self._store.search_patterns(query, limit=limit)
        if not hits:
            return ToolResult.success("no matching patterns")
        for p in hits:
            await self._store.touch_pattern(p.id)
        body = "\n\n".join(f"#{p.id} {p.render()}" for p in hits)
        return ToolResult.success(f"{len(hits)} match(es):\n\n{body}")


class ForgetPatternTool(BaseTool):
    name = "forget_pattern"
    description = "Delete a pattern by id. Use after recall_patterns to drop stale lessons."
    parameters = {
        "type": "object",
        "properties": {"id": {"type": "integer"}},
        "required": ["id"],
    }
    timeout_s = 5.0

    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    async def run(self, id: int) -> ToolResult:  # noqa: A002 - LLM-facing name
        ok = await self._store.delete_pattern(id)
        return ToolResult(ok=ok, content=f"deleted #{id}" if ok else f"no pattern #{id}")


class NoteFactTool(BaseTool):
    name = "note_fact"
    description = (
        "Persist a small key/value fact about the user, project, or environment. "
        "Useful for preferences ('preferred_editor=vscode') or state notes."
    )
    parameters = {
        "type": "object",
        "properties": {
            "key": {"type": "string"},
            "value": {"type": "string"},
        },
        "required": ["key", "value"],
    }
    timeout_s = 5.0

    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    async def run(self, key: str, value: str) -> ToolResult:
        await self._store.set_fact(key.strip(), value.strip())
        return ToolResult.success(f"noted {key} = {value}")


class GetFactTool(BaseTool):
    name = "get_fact"
    description = "Look up a previously noted fact by key, or list all if no key is given."
    parameters = {
        "type": "object",
        "properties": {"key": {"type": "string"}},
    }
    timeout_s = 5.0

    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    async def run(self, key: str | None = None) -> ToolResult:
        if key:
            v = await self._store.get_fact(key)
            return ToolResult(
                ok=v is not None,
                content=f"{key} = {v}" if v is not None else f"no fact stored for {key}",
            )
        facts = await self._store.all_facts()
        if not facts:
            return ToolResult.success("no facts stored yet")
        body = "\n".join(f"{k} = {v}" for k, v in facts.items())
        return ToolResult.success(body)


def tools(store: MemoryStore) -> list[Any]:
    return [
        RememberPatternTool(store),
        RecallPatternsTool(store),
        ForgetPatternTool(store),
        NoteFactTool(store),
        GetFactTool(store),
    ]
