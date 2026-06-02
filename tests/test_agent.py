"""End-to-end agent loop with a scripted fake provider.

Exercises the real turn(): system-prompt refresh, user append, compaction,
streaming, parallel tool dispatch, and the tool-result feedback — plus the
provider-error path. No network, no real model.
"""
from __future__ import annotations

from typing import Any, AsyncIterator

from krypton.core.agent import Agent
from krypton.memory.store import MemoryStore
from krypton.providers.base import Done, Message, StreamEvent, TextDelta, ToolCall, ToolCallDelta
from krypton.tools.base import BaseTool, ToolResult
from krypton.tools.registry import ToolRegistry


class EchoTool(BaseTool):
    name = "echo"
    description = "echo back the value"
    parameters = {"type": "object", "properties": {"v": {"type": "string"}}, "required": ["v"]}
    timeout_s = 5.0

    async def run(self, v: str) -> ToolResult:
        return ToolResult.success(f"echo:{v}")


class ScriptedProvider:
    """Yields a pre-baked list of StreamEvents per turn."""

    name = "fake"
    model = "fake-1"

    def __init__(self, turns: list[list[StreamEvent]]) -> None:
        self._turns = turns
        self._i = 0

    async def stream(self, messages: list[Message], tools: list[dict[str, Any]] | None = None,
                     *, temperature: float = 0.7, max_tokens: int | None = None) -> AsyncIterator[StreamEvent]:
        turn = self._turns[self._i]
        self._i += 1
        for ev in turn:
            yield ev

    async def aclose(self) -> None:
        pass


class BoomProvider:
    name = "boom"
    model = "x"

    async def stream(self, messages, tools=None, *, temperature=0.7, max_tokens=None):
        raise RuntimeError("network down")
        yield Done()  # pragma: no cover - makes this an async generator

    async def aclose(self) -> None:
        pass


async def test_turn_runs_tool_then_finishes(tmp_path):
    reg = ToolRegistry()
    reg.register(EchoTool())
    mem = MemoryStore(tmp_path / "m.db")
    prov = ScriptedProvider([
        [ToolCallDelta(ToolCall(id="c1", name="echo", arguments={"v": "hi"})), Done()],
        [TextDelta("all "), TextDelta("done"), Done()],
    ])
    agent = Agent(provider=prov, registry=reg, memory=mem)

    seen: list[tuple[str, bool]] = []

    async def on_tool(name: str, args: dict, res: ToolResult) -> None:
        seen.append((name, res.ok))

    try:
        result = await agent.turn("esegui", on_tool=on_tool)
        assert result.final_text == "all done"
        assert result.iterations == 2
        assert result.aborted is False
        assert seen == [("echo", True)]
        # the tool result is present in context and the pairing is valid
        msgs = agent.context.messages()
        tool_msgs = [m for m in msgs if m.role == "tool"]
        assert tool_msgs and "echo:hi" in tool_msgs[0].content
        call_ids = {tc.id for m in msgs if m.role == "assistant" for tc in m.tool_calls}
        assert all(m.tool_call_id in call_ids for m in tool_msgs)
    finally:
        await mem.aclose()


async def test_turn_streams_text_to_callback(tmp_path):
    reg = ToolRegistry()
    mem = MemoryStore(tmp_path / "m.db")
    prov = ScriptedProvider([[TextDelta("hello "), TextDelta("world"), Done()]])
    agent = Agent(provider=prov, registry=reg, memory=mem)

    chunks: list[str] = []

    async def on_text(t: str) -> None:
        chunks.append(t)

    try:
        result = await agent.turn("ciao", on_text=on_text)
        assert "".join(chunks) == "hello world"
        assert result.final_text == "hello world"
        assert result.iterations == 1
    finally:
        await mem.aclose()


async def test_turn_handles_provider_error(tmp_path):
    reg = ToolRegistry()
    mem = MemoryStore(tmp_path / "m.db")
    agent = Agent(provider=BoomProvider(), registry=reg, memory=mem)
    try:
        result = await agent.turn("hi")
        assert result.aborted is True
        assert result.error and "network down" in result.error
    finally:
        await mem.aclose()
