"""NVIDIA provider: requests must stay model-agnostic.

Regression guard for the "(no answer)" / conflict issue when swapping NIM
models: the provider must NOT force model-specific reasoning params onto every
request. By default it sends a plain OpenAI-compatible body; reasoning is opt-in
and only adds the *standard* `reasoning_effort` field.
"""
from __future__ import annotations

from krypton.providers.base import Message
from krypton.providers.nvidia import NvidiaProvider

_MSGS = [Message(role="user", content="hi")]


async def test_default_payload_has_no_reasoning_fields():
    p = NvidiaProvider(model="minimaxai/minimax-m3", api_key="dummy", reasoning_effort=None)
    try:
        pl = p._payload(_MSGS, None, 0.7, None)
        assert "reasoning_effort" not in pl
        assert "chat_template_kwargs" not in pl          # the field that broke models
        assert pl["model"] == "minimaxai/minimax-m3"
        assert pl["stream"] is True
    finally:
        await p.aclose()


async def test_none_effort_sends_nothing():
    p = NvidiaProvider(model="x/y", api_key="dummy", reasoning_effort="none")
    try:
        pl = p._payload(_MSGS, None, 0.7, None)
        assert "reasoning_effort" not in pl
        assert "chat_template_kwargs" not in pl
    finally:
        await p.aclose()


async def test_explicit_level_adds_only_standard_field():
    p = NvidiaProvider(model="nvidia/nemotron-3-super", api_key="dummy", reasoning_effort="high")
    try:
        pl = p._payload(_MSGS, None, 0.7, None)
        assert pl["reasoning_effort"] == "high"
        assert "chat_template_kwargs" not in pl          # never auto-injected
    finally:
        await p.aclose()


async def test_tools_passthrough():
    p = NvidiaProvider(model="x/y", api_key="dummy", reasoning_effort=None)
    try:
        tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
        pl = p._payload(_MSGS, tools, 0.5, 256)
        assert pl["tools"] == tools and pl["tool_choice"] == "auto"
        assert pl["max_tokens"] == 256 and pl["temperature"] == 0.5
    finally:
        await p.aclose()


async def test_extra_body_overrides_win():
    p = NvidiaProvider(model="x/y", api_key="dummy", reasoning_effort=None)
    p.extra_body = {"chat_template_kwargs": {"thinking": True}}
    try:
        pl = p._payload(_MSGS, None, 0.7, None)
        assert pl["chat_template_kwargs"] == {"thinking": True}  # opt-in only
    finally:
        await p.aclose()
