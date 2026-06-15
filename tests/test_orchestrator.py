"""Parallel specialist sub-agent orchestration (Hermes-style hub-and-spoke)."""
from __future__ import annotations

from krypton.config import settings
from krypton.core.prompt import build_subagent_prompt
from krypton.memory.store import MemoryStore
from krypton.providers.base import Done, TextDelta
from krypton.tools.orchestrator import SpawnAgentsTool
from krypton.tools.registry import ToolRegistry


class EchoProvider:
    """Answers each turn by echoing the worker's instructions. Stateless, so it's
    safe under the concurrent calls the orchestrator makes across workers."""

    name = "echo"
    model = "echo-1"

    async def stream(self, messages, tools=None, *, temperature=0.7, max_tokens=None):
        last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        yield TextDelta(f"done: {last_user.strip()}")
        yield Done()

    async def aclose(self):
        pass


async def test_spawn_runs_parallel_and_aggregates(tmp_path):
    mem = MemoryStore(tmp_path / "m.db")
    try:
        tool = SpawnAgentsTool(
            get_provider=lambda: EchoProvider(),
            memory=mem,
            make_worker_registry=ToolRegistry,
            worker_iterations=3,
        )
        res = await tool.run(tasks=[
            {"role": "researcher", "instructions": "research topic A"},
            {"role": "analyst", "instructions": "analyze dataset B"},
        ])
        assert res.ok
        assert "[researcher]" in res.content and "[analyst]" in res.content
        assert "research topic A" in res.content and "analyze dataset B" in res.content
        assert res.meta["succeeded"] == 2 and res.meta["workers"] == 2
    finally:
        await mem.aclose()


async def test_spawn_caps_workers(tmp_path):
    mem = MemoryStore(tmp_path / "m.db")
    try:
        tool = SpawnAgentsTool(
            get_provider=lambda: EchoProvider(), memory=mem,
            make_worker_registry=ToolRegistry, max_workers=2,
        )
        res = await tool.run(tasks=[{"role": f"r{i}", "instructions": f"t{i}"} for i in range(5)])
        assert res.content.count("### [") == 2
        assert "only the first 2" in res.content
    finally:
        await mem.aclose()


async def test_spawn_rejects_empty(tmp_path):
    mem = MemoryStore(tmp_path / "m.db")
    try:
        tool = SpawnAgentsTool(get_provider=lambda: EchoProvider(), memory=mem,
                               make_worker_registry=ToolRegistry)
        assert not (await tool.run(tasks=[])).ok
        assert not (await tool.run(tasks=[{"role": "x", "instructions": "   "}])).ok
    finally:
        await mem.aclose()


async def test_spawn_isolates_worker_failure(tmp_path):
    class FlakyProvider:
        name = "flaky"
        model = "f"

        def __init__(self, fail_marker):
            self._fail = fail_marker

        async def stream(self, messages, tools=None, *, temperature=0.7, max_tokens=None):
            last = next((m.content for m in reversed(messages) if m.role == "user"), "")
            if self._fail in last:
                raise RuntimeError("boom")
            yield TextDelta(f"ok: {last.strip()}")
            yield Done()

        async def aclose(self):
            pass

    mem = MemoryStore(tmp_path / "m.db")
    try:
        tool = SpawnAgentsTool(get_provider=lambda: FlakyProvider("BAD"), memory=mem,
                               make_worker_registry=ToolRegistry)
        res = await tool.run(tasks=[
            {"role": "good", "instructions": "fine task"},
            {"role": "bad", "instructions": "BAD task"},
        ])
        assert "ok: fine task" in res.content      # healthy worker still returned
        assert res.meta["succeeded"] == 1          # the crash was contained
        assert "error" in res.content.lower()
    finally:
        await mem.aclose()


def test_subagent_prompt_is_scoped_and_grounded():
    p = build_subagent_prompt(role="data_analyst", registry=ToolRegistry())
    assert "data_analyst" in p
    assert "WORKER" in p
    assert "CANNOT spawn" in p
    assert "REMOTE CLOUD SERVER" in p              # inherits the env self-model
    assert "HALLUCINATE" in p                       # grounding directive


async def test_build_agent_wires_orchestrator_without_recursion(tmp_path, monkeypatch):
    from krypton.bootstrap import _make_worker_registry, build_agent

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    agent = build_agent()
    try:
        assert agent.registry.get("spawn_agents") is not None      # orchestrator can delegate
        wreg = _make_worker_registry(agent.memory)
        assert wreg.get("spawn_agents") is None                    # workers cannot (no fork bombs)
        assert wreg.get("read_file") is not None                   # but have the work tools
    finally:
        await agent.aclose()
