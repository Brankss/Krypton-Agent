"""Tool protocol and shared result types.

A Tool is the smallest unit the agent can invoke. It is fully async,
declares its JSON schema for the LLM, and returns a structured
ToolResult so the rest of the system can compress / display / archive
its output uniformly.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable


class ToolError(Exception):
    """Raised inside a tool to signal a controlled failure.

    The agent sees the message and can recover. Uncaught exceptions are
    wrapped automatically by the registry.
    """


@dataclass(slots=True)
class ToolArtifact:
    """A file produced by a tool that can be forwarded to the user.

    `path` is absolute on disk; `mime` is best-effort; `label` is what
    the agent named it for human consumption.
    """
    path: str
    mime: str = "application/octet-stream"
    label: str = ""


@dataclass(slots=True)
class ToolResult:
    ok: bool
    content: str = ""
    artifacts: list[ToolArtifact] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    duration_ms: int = 0

    @classmethod
    def success(cls, content: str = "", **kw) -> "ToolResult":
        return cls(ok=True, content=content, **kw)

    @classmethod
    def failure(cls, message: str, **kw) -> "ToolResult":
        return cls(ok=False, content=message, **kw)


@runtime_checkable
class Tool(Protocol):
    """Async tool contract.

    Each tool MUST expose:
      - name: short snake_case identifier the LLM will call
      - description: 1–2 lines, action-oriented
      - parameters: JSON schema (dict) of the input
      - run(**kwargs): the actual async work
    """
    name: str
    description: str
    parameters: dict[str, Any]
    timeout_s: float

    async def run(self, **kwargs: Any) -> ToolResult: ...


# ---------------------------------------------------------------------------
# A small convenience base class for the common pattern of "I'm a function".
# Tools that want full control just implement the Protocol directly.
# ---------------------------------------------------------------------------


class BaseTool:
    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    timeout_s: float = 30.0

    async def run(self, **kwargs: Any) -> ToolResult:  # pragma: no cover - abstract
        raise NotImplementedError


async def run_with_timeout(
    fn: Callable[..., Awaitable[ToolResult]],
    timeout_s: float,
    /,
    **kwargs: Any,
) -> ToolResult:
    """Invoke a tool function with a hard timeout, timing it."""
    start = time.perf_counter()
    try:
        result = await asyncio.wait_for(fn(**kwargs), timeout=timeout_s)
    except asyncio.TimeoutError:
        result = ToolResult.failure(f"timeout after {timeout_s:.1f}s")
    except ToolError as e:
        result = ToolResult.failure(str(e))
    except Exception as e:  # noqa: BLE001 - tools must never crash the loop
        result = ToolResult.failure(f"{type(e).__name__}: {e}")
    result.duration_ms = int((time.perf_counter() - start) * 1000)
    return result
