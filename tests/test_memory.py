"""Memory store: atomic dedup-or-insert + facts.

The regression target is the TOCTOU between the old search-then-add path and
parallel tool dispatch: two identical `remember_pattern` calls firing
concurrently must NOT produce two rows. `store.remember()` runs the whole
check-then-act under the write lock, so they collapse to one.
"""
from __future__ import annotations

import asyncio

from krypton.memory.store import MemoryStore


async def test_remember_inserts_new(tmp_path):
    s = MemoryStore(tmp_path / "m.db")
    try:
        p, merged = await s.remember("recipe", "Use ripgrep", "rg is fast", ["search"])
        assert merged is False and p.id > 0
        pats = await s.list_patterns()
        assert len(pats) == 1 and pats[0].title == "Use ripgrep"
        assert "search" in pats[0].tags
    finally:
        await s.aclose()


async def test_remember_merges_similar_title(tmp_path):
    s = MemoryStore(tmp_path / "m.db")
    try:
        await s.remember("recipe", "Use ripgrep for fast search", "body one")
        _, merged = await s.remember("recipe", "Use ripgrep for fast search", "body two")
        assert merged is True
        pats = await s.list_patterns()
        assert len(pats) == 1
        assert "body one" in pats[0].body and "body two" in pats[0].body
    finally:
        await s.aclose()


async def test_remember_different_kind_not_merged(tmp_path):
    s = MemoryStore(tmp_path / "m.db")
    try:
        await s.remember("recipe", "Same exact title here", "a")
        await s.remember("known_error", "Same exact title here", "b")
        assert len(await s.list_patterns()) == 2
    finally:
        await s.aclose()


async def test_concurrent_remember_no_double_insert(tmp_path):
    # The headline race: parallel dispatch firing the same remember_pattern.
    s = MemoryStore(tmp_path / "m.db")
    try:
        title = "Always quote Windows paths that contain spaces"
        await asyncio.gather(*[
            s.remember("known_error", title, f"detail {i}") for i in range(12)
        ])
        assert len(await s.list_patterns()) == 1
    finally:
        await s.aclose()


async def test_facts_roundtrip(tmp_path):
    s = MemoryStore(tmp_path / "m.db")
    try:
        await s.set_fact("preferred_editor", "vscode")
        assert await s.get_fact("preferred_editor") == "vscode"
        await s.set_fact("preferred_editor", "neovim")  # upsert
        assert await s.get_fact("preferred_editor") == "neovim"
        assert (await s.all_facts())["preferred_editor"] == "neovim"
        assert await s.delete_fact("preferred_editor") is True
        assert await s.get_fact("preferred_editor") is None
    finally:
        await s.aclose()


async def test_search_patterns_ranks_by_token_hits(tmp_path):
    s = MemoryStore(tmp_path / "m.db")
    try:
        await s.remember("recipe", "ollama http timeout fix", "increase read_timeout")
        await s.remember("recipe", "telegram markdown escaping", "escape specials")
        hits = await s.search_patterns("ollama timeout", limit=5)
        assert hits and "ollama" in hits[0].title
    finally:
        await s.aclose()
