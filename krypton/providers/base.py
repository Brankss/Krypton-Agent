"""Provider-agnostic message + streaming types.

The agent loop only talks to `LLMProvider.stream(...)` and consumes a
sequence of `StreamEvent`s. Each concrete provider translates from its
own wire format into these primitives.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Literal, Protocol, runtime_checkable

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(slots=True)
class ToolCall:
    """A function call requested by the model."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class Message:
    """Canonical chat message used inside the agent.

    For role='tool' we attach `tool_call_id` so providers that need
    a strict request/response pairing (OpenRouter, etc.) can wire it.
    """
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None  # tool name when role='tool'


# ---- streaming events ------------------------------------------------------


@dataclass(slots=True)
class TextDelta:
    text: str


@dataclass(slots=True)
class ToolCallDelta:
    """Emitted once per tool call decided by the model (non-incremental).

    We accumulate provider-side and emit a complete ToolCall, because all
    three providers either give us tool calls atomically (Ollama) or as
    incremental JSON fragments we have to buffer anyway (OpenRouter).
    """
    call: ToolCall


@dataclass(slots=True)
class Done:
    """End-of-turn marker with the final usage info if available."""
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)


StreamEvent = TextDelta | ToolCallDelta | Done


@runtime_checkable
class LLMProvider(Protocol):
    name: str
    model: str

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamEvent]: ...

    async def aclose(self) -> None: ...
