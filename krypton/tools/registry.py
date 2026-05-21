"""Tool registry: register, list, dispatch.

  * Cached OpenAI-shaped schema (invalidated when the tool set changes).
  * Per-tool metrics (calls, errors, total_ms) — useful for /status & profiling.
  * Bounded-timeout dispatch via run_with_timeout.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from krypton.tools.base import Tool, ToolResult, run_with_timeout


@dataclass(slots=True)
class _Stat:
    calls: int = 0
    errors: int = 0
    total_ms: int = 0

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.calls if self.calls else 0.0


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._schema_cache: list[dict[str, Any]] | None = None
        self._stats: dict[str, _Stat] = {}

    # ----- registration -------------------------------------------------
    def register(self, tool: Tool) -> Tool:
        if not tool.name or not tool.name.isidentifier():
            raise ValueError(f"invalid tool name: {tool.name!r}")
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool
        self._stats[tool.name] = _Stat()
        self._schema_cache = None
        return tool

    def register_many(self, tools: Iterable[Tool]) -> None:
        for t in tools:
            self.register(t)

    # ----- introspection ------------------------------------------------
    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def stats(self) -> dict[str, _Stat]:
        return dict(self._stats)

    def as_openai_schema(self) -> list[dict[str, Any]]:
        """Cached: built once per registration set."""
        if self._schema_cache is not None:
            return self._schema_cache
        out: list[dict[str, Any]] = []
        for t in self._tools.values():
            out.append(
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": (t.description or "").strip(),
                        "parameters": t.parameters or {"type": "object", "properties": {}},
                    },
                }
            )
        self._schema_cache = out
        return out

    # ----- dispatch -----------------------------------------------------
    async def dispatch(self, name: str, arguments: dict[str, Any] | None) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult.failure(
                f"unknown tool {name!r}. Available: {', '.join(self.names())}"
            )
        args = arguments or {}
        if not isinstance(args, dict):
            return ToolResult.failure(f"arguments for {name} must be a JSON object")
        res = await run_with_timeout(tool.run, tool.timeout_s, **args)
        st = self._stats[name]
        st.calls += 1
        st.total_ms += res.duration_ms
        if not res.ok:
            st.errors += 1
        return res


default_registry = ToolRegistry()
