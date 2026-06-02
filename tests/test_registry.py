"""Tool registry: dispatch, timeout, error wrapping, schema cache, clear."""
from __future__ import annotations

import asyncio

import pytest

from krypton.tools.base import BaseTool, ToolResult
from krypton.tools.registry import ToolRegistry


class EchoTool(BaseTool):
    name = "echo"
    description = "echo back"
    parameters = {"type": "object", "properties": {"v": {"type": "string"}}, "required": ["v"]}
    timeout_s = 5.0

    async def run(self, v: str) -> ToolResult:
        return ToolResult.success(v)


class BoomTool(BaseTool):
    name = "boom"
    description = "always raises"
    timeout_s = 5.0

    async def run(self) -> ToolResult:
        raise RuntimeError("kaboom")


class SlowTool(BaseTool):
    name = "slow"
    description = "exceeds its timeout"
    timeout_s = 0.05

    async def run(self) -> ToolResult:
        await asyncio.sleep(1.0)
        return ToolResult.success("done")


async def test_dispatch_ok_and_stats():
    r = ToolRegistry()
    r.register(EchoTool())
    res = await r.dispatch("echo", {"v": "hi"})
    assert res.ok and res.content == "hi"
    assert r.stats()["echo"].calls == 1
    assert r.stats()["echo"].errors == 0


async def test_unknown_tool_is_failure():
    r = ToolRegistry()
    res = await r.dispatch("nope", {})
    assert not res.ok and "unknown tool" in res.content


async def test_tool_exception_becomes_failure():
    r = ToolRegistry()
    r.register(BoomTool())
    res = await r.dispatch("boom", {})
    assert not res.ok and "kaboom" in res.content
    assert r.stats()["boom"].errors == 1


async def test_timeout_is_failure():
    r = ToolRegistry()
    r.register(SlowTool())
    res = await r.dispatch("slow", {})
    assert not res.ok and "timeout" in res.content


async def test_non_dict_arguments_rejected():
    r = ToolRegistry()
    r.register(EchoTool())
    res = await r.dispatch("echo", ["not", "a", "dict"])  # type: ignore[arg-type]
    assert not res.ok and "JSON object" in res.content


def test_schema_cache_identity_and_invalidation():
    r = ToolRegistry()
    r.register(EchoTool())
    s1 = r.as_openai_schema()
    s2 = r.as_openai_schema()
    assert s1 is s2  # cached
    r.register(BoomTool())
    s3 = r.as_openai_schema()
    assert s3 is not s1 and len(s3) == 2  # invalidated on register


def test_clear_resets_everything():
    r = ToolRegistry()
    r.register(EchoTool())
    r.clear()
    assert r.names() == []
    assert r.as_openai_schema() == []


def test_duplicate_registration_rejected():
    r = ToolRegistry()
    r.register(EchoTool())
    with pytest.raises(ValueError):
        r.register(EchoTool())


def test_invalid_tool_name_rejected():
    bad = EchoTool()
    bad.name = "has spaces"
    r = ToolRegistry()
    with pytest.raises(ValueError):
        r.register(bad)
