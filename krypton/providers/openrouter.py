"""OpenRouter provider (OpenAI-compatible Server-Sent Events stream).

OpenRouter emits incremental tool-call fragments: name arrives early,
arguments stream in as JSON-text chunks. We buffer per-index and emit
a single ToolCallDelta when the call is complete.
"""
from __future__ import annotations

import json
import uuid
from typing import Any, AsyncIterator

import httpx

from krypton.providers._msg import messages_to_openai
from krypton.providers.base import (
    Done,
    LLMProvider,
    Message,
    StreamEvent,
    TextDelta,
    ToolCall,
    ToolCallDelta,
)


class OpenRouterProvider(LLMProvider):
    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        connect_timeout: float = 10.0,
        read_timeout: float = 600.0,
        referer: str = "https://krypton.local",
        title: str = "Krypton Agent",
    ) -> None:
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required to use the openrouter provider")
        self.name = "openrouter"
        self.model = model
        self._base = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": referer,
                "X-Title": title,
            },
            timeout=httpx.Timeout(read_timeout, connect=connect_timeout),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamEvent]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages_to_openai(messages),
            "stream": True,
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        # buffer for partial tool calls keyed by index
        buf: dict[int, dict[str, Any]] = {}
        finish_reason = "stop"
        usage: dict[str, int] = {}

        url = f"{self._base}/chat/completions"
        async with self._client.stream("POST", url, json=payload) as resp:
            if resp.status_code >= 400:
                body = (await resp.aread()).decode("utf-8", "replace")
                raise RuntimeError(f"OpenRouter HTTP {resp.status_code}: {body[:500]}")

            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                choice = (chunk.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                if (txt := delta.get("content")):
                    yield TextDelta(txt)

                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    slot = buf.setdefault(idx, {"id": "", "name": "", "args": ""})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["args"] += fn["arguments"]

                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]

                if chunk.get("usage"):
                    u = chunk["usage"]
                    usage = {
                        "prompt_tokens": int(u.get("prompt_tokens") or 0),
                        "completion_tokens": int(u.get("completion_tokens") or 0),
                    }

        # flush buffered tool calls
        for slot in buf.values():
            if not slot["name"]:
                continue
            try:
                args = json.loads(slot["args"] or "{}")
            except json.JSONDecodeError:
                args = {"_raw": slot["args"]}
            yield ToolCallDelta(
                ToolCall(
                    id=slot["id"] or f"call_{uuid.uuid4().hex[:12]}",
                    name=slot["name"],
                    arguments=args,
                )
            )

        yield Done(finish_reason=finish_reason, usage=usage)
