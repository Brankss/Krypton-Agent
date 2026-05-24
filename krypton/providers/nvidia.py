"""NVIDIA NIM provider (OpenAI-compatible SSE).

NVIDIA's hosted inference endpoint at https://integrate.api.nvidia.com/v1
speaks the OpenAI Chat Completions wire format. Auth via Bearer token,
streaming via Server-Sent Events.

Per-model knobs (thinking, reasoning_effort, etc) are routed via
`extra_body` which NIM forwards to the underlying inference engine. We
expose a single `thinking: bool` toggle that gets translated to
`chat_template_kwargs.thinking` — the convention used by Kimi K2-Thinking,
Qwen3, Nemotron-Nano-9B-V2 and most other thinking-capable NIM models.

If you need finer control (e.g. `reasoning_effort` for gpt-oss models)
set `provider.extra_body` directly from outside.
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


class NvidiaProvider(LLMProvider):
    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        thinking: bool = True,
        connect_timeout: float = 10.0,
        read_timeout: float = 600.0,
    ) -> None:
        if not api_key:
            raise RuntimeError("NVIDIA_API_KEY is required to use the nvidia provider")
        self.name = "nvidia"
        self.model = model
        self.thinking = thinking
        # Free-form per-request overrides. /thinking writes here; users can poke too.
        self.extra_body: dict[str, Any] = {}
        self._base = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
            timeout=httpx.Timeout(read_timeout, connect=connect_timeout),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------
    def _payload(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        temperature: float,
        max_tokens: int | None,
    ) -> dict[str, Any]:
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

        # Thinking toggle — convention used by Kimi K2-Thinking / Qwen3 / Nemotron
        # via the chat template. Harmless for models that ignore unknown kwargs.
        ctk = {"thinking": bool(self.thinking)}
        if self.extra_body:
            # explicit extra_body wins
            merged_ctk = dict(ctk)
            merged_ctk.update(self.extra_body.get("chat_template_kwargs") or {})
            payload.update({k: v for k, v in self.extra_body.items() if k != "chat_template_kwargs"})
            payload["chat_template_kwargs"] = merged_ctk
        else:
            payload["chat_template_kwargs"] = ctk

        return payload

    # ------------------------------------------------------------------
    async def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamEvent]:
        payload = self._payload(messages, tools, temperature, max_tokens)

        # buffer for partial tool calls keyed by index
        buf: dict[int, dict[str, Any]] = {}
        finish_reason = "stop"
        usage: dict[str, int] = {}
        # Track whether we've emitted the visible thinking tag yet
        in_reasoning = False

        url = f"{self._base}/chat/completions"
        async with self._client.stream("POST", url, json=payload) as resp:
            if resp.status_code >= 400:
                body = (await resp.aread()).decode("utf-8", "replace")
                raise RuntimeError(f"NVIDIA HTTP {resp.status_code}: {body[:500]}")

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

                # Some NIM models stream a separate `reasoning_content` field
                # (DeepSeek-R1 / Nemotron). Surface it inline so the UI can see
                # the chain-of-thought when /thinking is on, but mark it.
                if (rc := delta.get("reasoning_content")):
                    if not in_reasoning:
                        yield TextDelta("\n_[thinking]_\n")
                        in_reasoning = True
                    yield TextDelta(rc)

                if (txt := delta.get("content")):
                    if in_reasoning:
                        yield TextDelta("\n_[/thinking]_\n\n")
                        in_reasoning = False
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
