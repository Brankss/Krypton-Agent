"""Shared message <-> wire format converters reused by Ollama / OpenRouter.

Both providers accept the OpenAI-shaped chat format these days, so we
centralize the encoding once and let each subclass tweak edges.
"""
from __future__ import annotations

import json
from typing import Any

from krypton.providers.base import Message


def messages_to_openai(messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "tool":
            out.append(
                {
                    "role": "tool",
                    "content": m.content,
                    "tool_call_id": m.tool_call_id or "",
                    "name": m.name or "",
                }
            )
            continue
        d: dict[str, Any] = {"role": m.role, "content": m.content or ""}
        if m.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                    },
                }
                for tc in m.tool_calls
            ]
        out.append(d)
    return out


def messages_to_ollama(messages: list[Message]) -> list[dict[str, Any]]:
    """Ollama expects nearly the same shape but `arguments` is a dict, not a JSON string."""
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "tool":
            out.append({"role": "tool", "content": m.content, "name": m.name or ""})
            continue
        d: dict[str, Any] = {"role": m.role, "content": m.content or ""}
        if m.tool_calls:
            d["tool_calls"] = [
                {
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments,
                    }
                }
                for tc in m.tool_calls
            ]
        out.append(d)
    return out
