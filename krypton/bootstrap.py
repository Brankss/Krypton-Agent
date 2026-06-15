"""Wire together provider + tools + memory + agent.

Single place that knows how to assemble a ready-to-use Agent. Both the
CLI and the Telegram interface call this so they stay perfectly in sync.
"""
from __future__ import annotations

from krypton.config import ProviderName, settings
from krypton.core.agent import Agent
from krypton.core.context import ConversationContext
from krypton.memory.store import MemoryStore
from krypton.providers import build_provider
from krypton.tools.base import Tool
from krypton.tools.files import tools as file_tools
from krypton.tools.filesystem import tools as fs_tools
from krypton.tools.memory import tools as memory_tools
from krypton.tools.shell import tools as shell_tools
from krypton.tools.web import tools as web_tools


def _worker_tools(memory: MemoryStore) -> list[Tool]:
    """The 'work' toolset shared by the main agent and every spawned worker:
    filesystem, files, web, shell/python, memory. NOT orchestration or chat
    tools (spawn_agents / send_file / schedule) — those are orchestrator-only."""
    out: list[Tool] = []
    out += fs_tools()
    out += file_tools()
    out += web_tools()
    out += shell_tools()
    out += memory_tools(memory)
    return out


def _make_worker_registry(memory: MemoryStore):
    from krypton.tools.registry import ToolRegistry
    reg = ToolRegistry()
    reg.register_many(_worker_tools(memory))
    return reg


def build_agent(
    provider_name: ProviderName | None = None,
    *,
    interface: str = "cli",
    extra_tools: list[Tool] | None = None,
) -> Agent:
    # Registries are stateful — build fresh per session to avoid duplicates
    from krypton.tools.registry import ToolRegistry
    registry = ToolRegistry()

    memory = MemoryStore(settings.data_dir / "krypton.db")

    registry.register_many(_worker_tools(memory))
    if extra_tools:
        registry.register_many(extra_tools)

    provider = build_provider(provider_name)
    ctx = ConversationContext(target_budget=settings.context_budget)

    agent = Agent(
        provider=provider,
        registry=registry,
        memory=memory,
        context=ctx,
        interface=interface,
    )

    # Orchestration: let the main agent spawn parallel specialist workers.
    # Registered post-construction so it reads the LIVE provider (which /provider
    # may swap) and builds a fresh scoped worker registry per spawn.
    from krypton.tools.orchestrator import SpawnAgentsTool
    agent.registry.register(
        SpawnAgentsTool(
            get_provider=lambda: agent.provider,
            memory=memory,
            make_worker_registry=lambda: _make_worker_registry(memory),
        )
    )
    return agent
