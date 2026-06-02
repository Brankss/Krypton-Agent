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

    registry.register_many(fs_tools())
    registry.register_many(file_tools())
    registry.register_many(web_tools())
    registry.register_many(shell_tools())
    registry.register_many(memory_tools(memory))
    if extra_tools:
        registry.register_many(extra_tools)

    provider = build_provider(provider_name)
    ctx = ConversationContext(target_budget=settings.context_budget)

    return Agent(
        provider=provider,
        registry=registry,
        memory=memory,
        context=ctx,
        interface=interface,
    )
