"""Ollama provider — supports both a local daemon and Ollama Cloud.

The wire protocol is identical; only host + optional bearer auth differ.

Thinking support: Ollama 0.7+ exposes a `think` parameter on /api/chat
that toggles chain-of-thought for capable models (qwen3, deepseek-r1,
gpt-oss, etc). We translate the cross-provider `reasoning_effort` knob
to Ollama's wire format:

  None       → don't send `think` (model default)
  "none"     → think: false
  "low" etc  → think: true   (bool is the universally-safe encoding;
                              recent Ollama also accepts string levels
                              for some models)

The model's `thinking` response field is silently consumed — we only
emit visible `content`. Inline <think>...</think> blocks (some models
bake reasoning into the content channel) are stripped via the shared
strip_think state machine.
"""
from __future__ import annotations

import json
import uuid
from typing import Any, AsyncIterator

import httpx

from krypton.providers._msg import messages_to_ollama
from krypton.providers._thinking import ReasoningEffort, normalize_effort, strip_think
from krypton.providers.base import (
    Done,
    LLMProvider,
    Message,
    StreamEvent,
    TextDelta,
    ToolCall,
    ToolCallDelta,
)


class OllamaProvider(LLMProvider):
    """Streaming Ollama /api/chat client with tool-calling."""

    def __init__(
        self,
        *,
        name: str,
        host: str,
        model: str,
        api_key: str = "",
        reasoning_effort: ReasoningEffort | None = None,
        connect_timeout: float = 10.0,
        read_timeout: float = 600.0,
    ) -> None:
        self.name = name
        self.model = model
        self._reasoning_effort: ReasoningEffort | None = None
        self.reasoning_effort = reasoning_effort  # type: ignore[assignment]
        self._host = host.rstrip("/")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            headers=headers,
            timeout=httpx.Timeout(read_timeout, connect=connect_timeout),
        )

    # --- reasoning_effort knob (same interface as NvidiaProvider) -----
    @property
    def reasoning_effort(self) -> ReasoningEffort | None:
        return self._reasoning_effort

    @reasoning_effort.setter
    def reasoning_effort(self, value: str | None) -> None:
        self._reasoning_effort = normalize_effort(value)

    @property
    def thinking(self) -> bool:
        return self._reasoning_effort not in (None, "none")

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
            "messages": messages_to_ollama(messages),
            "stream": True,
            "keep_alive": "30m",  # keep the model warm in VRAM between turns
            "options": {"temperature": temperature},
        }
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        if tools:
            payload["tools"] = tools

        # Translate reasoning_effort -> Ollama's `think` parameter.
        eff = self._reasoning_effort
        if eff == "none":
            payload["think"] = False
        elif eff in ("low", "medium", "high"):
            payload["think"] = True  # safe bool encoding; works on all Ollama versions

        # State for stripping inline <think>...</think> from the content channel.
        in_think = False
        carry = ""

        url = f"{self._host}/api/chat"
        async with self._client.stream("POST", url, json=payload) as resp:
            if resp.status_code >= 400:
                body = (await resp.aread()).decode("utf-8", "replace")
                raise RuntimeError(f"Ollama HTTP {resp.status_code}: {body[:500]}")

            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                msg = chunk.get("message") or {}

                # Silently consume the dedicated thinking channel.
                _ = msg.get("thinking")

                text = msg.get("content") or ""
                if text:
                    clean, in_think, carry = strip_think(text, in_think, carry)
                    if clean:
                        yield TextDelta(clean)

                for tc in msg.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    args = fn.get("arguments") or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {"_raw": args}
                    yield ToolCallDelta(
                        ToolCall(
                            id=tc.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                            name=fn.get("name") or "",
                            arguments=args,
                        )
                    )

                if chunk.get("done"):
                    if carry and not in_think:
                        yield TextDelta(carry)
                    yield Done(
                        finish_reason=chunk.get("done_reason") or "stop",
                        usage={
                            "prompt_tokens": int(chunk.get("prompt_eval_count") or 0),
                            "completion_tokens": int(chunk.get("eval_count") or 0),
                        },
                    )
                    return
