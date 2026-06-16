"""Durable runtime config (prefs): provider / model / reasoning survive restarts."""
from __future__ import annotations

from krypton.bootstrap import build_agent
from krypton.config import settings
from krypton.memory.store import MemoryStore


async def test_prefs_crud_and_sync_read(tmp_path):
    s = MemoryStore(tmp_path / "m.db")
    try:
        assert await s.get_pref("x") is None
        await s.set_pref("x", "1")
        assert await s.get_pref("x") == "1"
        await s.set_pref("x", "2")  # upsert
        assert await s.get_pref("x") == "2"
        await s.set_pref("y", "")
        assert await s.all_prefs() == {"x": "2", "y": ""}
        assert s.load_prefs_sync() == {"x": "2", "y": ""}  # the startup path
        assert await s.delete_pref("x") is True
        assert await s.get_pref("x") is None
    finally:
        await s.aclose()


async def test_build_agent_applies_saved_provider_and_model(tmp_path, monkeypatch):
    seed = MemoryStore(tmp_path / "krypton.db")
    await seed.set_pref("provider", "ollama_local")
    await seed.set_pref("model", "my-custom-model:7b")
    await seed.aclose()

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    agent = build_agent()
    try:
        assert agent.provider.name == "ollama_local"
        assert agent.provider.model == "my-custom-model:7b"
    finally:
        await agent.aclose()


async def test_build_agent_applies_saved_reasoning(tmp_path, monkeypatch):
    seed = MemoryStore(tmp_path / "krypton.db")
    await seed.set_pref("provider", "ollama_local")
    await seed.set_pref("reasoning", "low")
    await seed.aclose()

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    agent = build_agent()
    try:
        assert agent.provider.reasoning_effort == "low"  # type: ignore[attr-defined]
    finally:
        await agent.aclose()


async def test_build_agent_survives_stale_provider_pref(tmp_path, monkeypatch):
    seed = MemoryStore(tmp_path / "krypton.db")
    await seed.set_pref("provider", "bogus_provider")  # e.g. removed in an update
    await seed.aclose()

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    agent = build_agent()  # must NOT crash the bot
    try:
        assert agent.provider.name == settings.provider  # fell back to the .env default
    finally:
        await agent.aclose()


async def test_model_pref_not_applied_across_providers(tmp_path, monkeypatch):
    # model saved under one provider must not leak onto a different one
    seed = MemoryStore(tmp_path / "krypton.db")
    await seed.set_pref("provider", "openrouter")     # different from what we'll force
    await seed.set_pref("model", "anthropic/claude-x")
    await seed.aclose()

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    agent = build_agent("ollama_local")  # explicit arg wins over the saved provider
    try:
        assert agent.provider.name == "ollama_local"
        assert agent.provider.model != "anthropic/claude-x"  # not leaked
    finally:
        await agent.aclose()
