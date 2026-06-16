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


def _build_provider_with_prefs(provider_name: ProviderName | None, prefs: dict[str, str]):
    """Resolve the provider with precedence: explicit arg > saved pref > .env.
    Then apply the user's persisted model / reasoning so live choices survive
    restarts and auto-deploys. Stale/invalid prefs fall back gracefully."""
    pref_provider = prefs.get("provider")
    chosen: str = provider_name or pref_provider or settings.provider
    try:
        provider = build_provider(chosen)  # may raise (unknown provider / missing key)
    except Exception:  # noqa: BLE001
        chosen = settings.provider
        provider = build_provider(settings.provider)

    # model pref applies only when the active provider matches the one it was
    # saved under (commands keep provider+model paired).
    if prefs.get("model") and chosen == pref_provider:
        provider.model = prefs["model"]

    reasoning = prefs.get("reasoning")
    if reasoning is not None and hasattr(provider, "reasoning_effort"):
        try:
            provider.reasoning_effort = reasoning  # validates; "" -> off
        except Exception:  # noqa: BLE001
            pass
    return provider


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

    # Apply the user's persisted provider/model/reasoning over .env defaults so
    # live /provider /model /reasoning choices survive restarts and auto-deploys.
    provider = _build_provider_with_prefs(provider_name, memory.load_prefs_sync())
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
