"""NVIDIA NIM provider (OpenAI-compatible SSE).

NVIDIA's hosted inference endpoint at https://integrate.api.nvidia.com/v1
speaks the OpenAI Chat Completions wire format. Auth via Bearer token,
streaming via Server-Sent Events.

Per-model knobs (thinking, reasoning_effort, etc) are routed via the
payload directly — NIM accepts OpenAI-style `reasoning_effort` plus
arbitrary `chat_template_kwargs` that get forwarded to the inference
engine. We expose a single `reasoning_effort` knob:

  "none"   → reasoning disabled
  "low"    → minimal reasoning
  "medium" → balanced
  "high"   → maximum reasoning depth
  None     → model default

This works on Nemotron-3-Super / Nemotron-Nano-9B-V2 / gpt-oss-120b and
most other reasoning-capable NIM models. For Kimi K2.6 / Qwen3 which use
the `thinking` boolean instead, we translate automatically:
"none" → thinking=false, anything else → thinking=true.
"""
from __future__ import annotations

import json
import uuid
from typing import Any, AsyncIterator

import httpx

from krypton.providers._msg import messages_to_openai
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


class NvidiaProvider(LLMProvider):
    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        reasoning_effort: ReasoningEffort | None = "low",
        connect_timeout: float = 10.0,
        read_timeout: float = 600.0,
    ) -> None:
        if not api_key:
            raise RuntimeError("NVIDIA_API_KEY is required to use the nvidia provider")
        self.name = "nvidia"
        self.model = model
        self._reasoning_effort: ReasoningEffort | None = None
        self.reasoning_effort = reasoning_effort  # type: ignore[assignment]
        # Free-form per-request overrides. Users can set this from outside if
        # they want to inject arbitrary NIM-specific parameters.
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

    # --- reasoning_effort property with validation -------------------
    @property
    def reasoning_effort(self) -> ReasoningEffort | None:
        return self._reasoning_effort

    @reasoning_effort.setter
    def reasoning_effort(self, value: str | None) -> None:
        self._reasoning_effort = normalize_effort(value)

    # --- backwards-compat alias --------------------------------------
    @property
    def thinking(self) -> bool:
        """True if any reasoning is enabled (any level except 'none')."""
        return self._reasoning_effort not in (None, "none")

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

        # --- reasoning controls (OPT-IN, model-agnostic) ---
        # By default we send a PLAIN OpenAI-compatible request that works with
        # ANY NIM model (Nemotron, Kimi, MiniMax, DeepSeek, GLM, ...). Only when
        # an explicit effort level is set do we add the *standard* reasoning_effort
        # field. We deliberately do NOT auto-inject the non-standard
        # `chat_template_kwargs.thinking` (a Kimi/Qwen3 convention): on models
        # that don't expect it, it makes them reject the request or dump the whole
        # answer into the hidden reasoning channel — which surfaces as "(no answer)".
        # Power users who need it for a specific model can still pass it via extra_body.
        eff = self._reasoning_effort
        if eff in ("low", "medium", "high"):
            payload["reasoning_effort"] = eff

        # explicit per-request/model overrides win
        if self.extra_body:
            payload.update(self.extra_body)

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

        # State for stripping <think>...</think> blocks that some Nemotron /
        # DeepSeek variants embed inside the regular `content` field.
        in_inline_think = False
        content_carry = ""
        emitted_content = False
        reasoning_buf: list[str] = []

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

                # `reasoning_content` is the model's private chain-of-thought
                # (DeepSeek-R1 / Nemotron / MiniMax ...). We don't show it, but we
                # keep it as a fallback: some models put EVERYTHING here and leave
                # `content` empty, which would otherwise surface as "(no answer)".
                if (rc := delta.get("reasoning_content")):
                    reasoning_buf.append(rc)

                # Content channel — strip inline <think>...</think> blocks
                # before forwarding. Carry partial tokens across chunks so we
                # don't accidentally split a tag boundary.
                if (txt := delta.get("content")):
                    clean, in_inline_think, content_carry = strip_think(
                        txt, in_inline_think, content_carry
                    )
                    if clean:
                        emitted_content = True
                        yield TextDelta(clean)

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

        # flush any trailing carry that wasn't a tag prefix after all
        if content_carry and not in_inline_think:
            emitted_content = True
            yield TextDelta(content_carry)

        # Fallback: the model produced no visible content and no tool calls but
        # did reason — surface the reasoning so the turn is never "(no answer)".
        if not emitted_content and not buf and reasoning_buf:
            yield TextDelta("".join(reasoning_buf).strip())

        yield Done(finish_reason=finish_reason, usage=usage)
