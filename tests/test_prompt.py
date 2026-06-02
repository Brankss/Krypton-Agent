"""System-prompt builder: imperative detection + static-portion caching."""
from __future__ import annotations

from krypton.core.prompt import _is_imperative, _static_prompt, build_system_prompt
from krypton.tools.base import BaseTool, ToolResult
from krypton.tools.registry import ToolRegistry


class DummyTool(BaseTool):
    name = "dummy"
    description = "does a thing\nsecond line should not appear"
    timeout_s = 5.0

    async def run(self) -> ToolResult:
        return ToolResult.success("")


def test_imperative_detection_positive():
    assert _is_imperative("mandami il file")
    assert _is_imperative("procedi")
    assert _is_imperative("esegui lo script ora")
    assert _is_imperative("cerca i log di ieri")


def test_imperative_detection_negative():
    assert not _is_imperative("ciao, come stai oggi?")
    assert not _is_imperative("")
    assert not _is_imperative(None)


def test_static_prompt_is_cached_by_toolset():
    r = ToolRegistry()
    r.register(DummyTool())
    a = _static_prompt(r)
    b = _static_prompt(r)
    assert a is b  # cached by tool-name signature
    assert "dummy: does a thing" in a
    assert "second line should not appear" not in a  # only the first description line


def test_build_prompt_appends_imperative_block():
    r = ToolRegistry()
    r.register(DummyTool())
    p = build_system_prompt(registry=r, last_user_text="mandami il log")
    assert "This turn" in p
    p2 = build_system_prompt(registry=r, last_user_text="che ore sono?")
    assert "This turn" not in p2


def test_build_prompt_includes_facts_and_patterns():
    from krypton.memory.store import Pattern

    r = ToolRegistry()
    r.register(DummyTool())
    pat = Pattern(id=1, kind="recipe", title="Quote paths", body="use quotes", tags=["shell"])
    p = build_system_prompt(
        registry=r,
        facts={"editor": "vscode"},
        patterns=[pat],
        last_user_text="hello",
    )
    assert "editor = vscode" in p
    assert "Quote paths" in p
