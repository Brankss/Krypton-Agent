"""Agent main loop.

`Agent.turn(user_text)` is the public entry point. It:
  1. Refreshes the system prompt with relevant memory hits for the user's text.
  2. Appends the user message.
  3. Streams from the provider, capturing tool calls.
  4. Dispatches tool calls IN PARALLEL.
  5. Loops until the model emits a turn with no tool calls (or we hit max_iter).

Callers get a `TurnResult` with:
  * final_text          — what the assistant said at the end
  * artifacts           — files the tools produced (forwarded by interfaces)
  * stream_iter         — optional async iterator for live display
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Callable, Awaitable

from krypton.config import settings
from krypton.core.context import ConversationContext
from krypton.core.prompt import build_system_prompt
from krypton.memory.store import MemoryStore
from krypton.providers.base import (
    Done,
    LLMProvider,
    Message,
    TextDelta,
    ToolCall,
    ToolCallDelta,
)
from krypton.tools.base import ToolArtifact, ToolResult
from krypton.tools.registry import ToolRegistry


# Callback signature for streaming UI: receives text deltas.
StreamCallback = Callable[[str], Awaitable[None]] | None
# Callback signature for tool-call notifications.
ToolCallback = Callable[[str, dict, ToolResult], Awaitable[None]] | None


@dataclass(slots=True)
class TurnResult:
    final_text: str = ""
    artifacts: list[ToolArtifact] = field(default_factory=list)
    iterations: int = 0
    aborted: bool = False
    error: str | None = None


class Agent:
    def __init__(
        self,
        *,
        provider: LLMProvider,
        registry: ToolRegistry,
        memory: MemoryStore,
        context: ConversationContext | None = None,
        interface: str = "cli",
    ) -> None:
        self.provider = provider
        self.registry = registry
        self.memory = memory
        self.context = context or ConversationContext(target_budget=settings.context_budget)
        self.interface = interface

    # ------------------------------------------------------------------
    async def _refresh_system_prompt(self, user_text: str) -> None:
        patterns = await self.memory.search_patterns(user_text, limit=6) if user_text else []
        facts = await self.memory.all_facts()
        prompt = build_system_prompt(
            registry=self.registry,
            patterns=patterns,
            facts=facts,
            provider_label=self.provider.name,
            model_label=self.provider.model,
            interface=self.interface,
            last_user_text=user_text,
        )
        self.context.set_system(prompt)

    # ------------------------------------------------------------------
    async def turn(
        self,
        user_text: str,
        *,
        on_text: StreamCallback = None,
        on_tool: ToolCallback = None,
    ) -> TurnResult:
        await self._refresh_system_prompt(user_text)
        self.context.add_user(user_text, pinned=True)

        result = TurnResult()
        for it in range(1, settings.max_iterations + 1):
            result.iterations = it
            self.context.compact()

            text_buf: list[str] = []
            pending_calls: list[ToolCall] = []
            usage: dict[str, int] = {}
            finish_reason = "stop"

            try:
                async for ev in self.provider.stream(
                    self.context.messages(),
                    tools=self.registry.as_openai_schema() or None,
                ):
                    if isinstance(ev, TextDelta):
                        text_buf.append(ev.text)
                        if on_text:
                            await on_text(ev.text)
                    elif isinstance(ev, ToolCallDelta):
                        pending_calls.append(ev.call)
                    elif isinstance(ev, Done):
                        finish_reason = ev.finish_reason
                        usage = ev.usage
            except Exception as e:  # noqa: BLE001
                result.error = f"provider error: {type(e).__name__}: {e}"
                result.aborted = True
                return result

            assistant_text = "".join(text_buf).strip()
            assistant_msg = Message(
                role="assistant",
                content=assistant_text,
                tool_calls=list(pending_calls),
            )
            self.context.add_assistant(assistant_msg)

            if not pending_calls:
                result.final_text = assistant_text
                _ = (finish_reason, usage)  # reserved for future telemetry
                return result

            # ----- dispatch tools in parallel ------------------------
            async def _dispatch(call: ToolCall) -> tuple[ToolCall, ToolResult]:
                res = await self.registry.dispatch(call.name, call.arguments)
                return call, res

            outcomes = await asyncio.gather(*[_dispatch(c) for c in pending_calls])
            for call, res in outcomes:
                if on_tool:
                    await on_tool(call.name, call.arguments, res)
                if res.artifacts:
                    result.artifacts.extend(res.artifacts)
                self.context.add_tool_result(
                    tool_call_id=call.id,
                    name=call.name,
                    content=_render_tool_result(res),
                )

        # Iteration cap hit. If the last entry is an assistant with unanswered
        # tool_calls, drop those tool_calls so the next turn's context isn't
        # structurally broken (a tool_call without a matching tool response).
        self.context.drop_trailing_unanswered_tool_calls()
        result.final_text = "(max iterations reached without final answer)"
        result.aborted = True
        return result

    async def aclose(self) -> None:
        await self.provider.aclose()
        await self.memory.aclose()


_TOOL_RESULT_CHAR_CAP = 16_000


def _render_tool_result(r: ToolResult) -> str:
    head = f"[ok={r.ok} took={r.duration_ms}ms]"
    body = r.content or ""
    if len(body) > _TOOL_RESULT_CHAR_CAP:
        keep = _TOOL_RESULT_CHAR_CAP // 2
        body = (
            body[:keep]
            + f"\n\n... [{len(body) - _TOOL_RESULT_CHAR_CAP} chars elided "
            f"— tool produced too much; rerun with stricter limits if you need this] ...\n\n"
            + body[-keep:]
        )
    if r.artifacts:
        body += "\nartifacts: " + ", ".join(a.path for a in r.artifacts)
    return f"{head}\n{body}".strip()
