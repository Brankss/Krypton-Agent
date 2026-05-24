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
from typing import Any, AsyncIterator, Literal

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


ReasoningEffort = Literal["none", "low", "medium", "high"]
_VALID_EFFORTS = {"none", "low", "medium", "high"}


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
        if value is None:
            self._reasoning_effort = None
            return
        v = value.strip().lower()
        if v not in _VALID_EFFORTS:
            raise ValueError(
                f"reasoning_effort must be one of {sorted(_VALID_EFFORTS)} or None, got {value!r}"
            )
        self._reasoning_effort = v  # type: ignore[assignment]

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

        # --- reasoning controls ---
        # We send BOTH conventions so a single setting works across the
        # heterogeneous NIM catalog:
        #   * reasoning_effort=<low|medium|high>   (Nemotron, gpt-oss)
        #   * chat_template_kwargs.thinking=bool    (Kimi K2.6, Qwen3)
        # Models ignore unknown fields, so this is safe.
        eff = self._reasoning_effort
        if eff is not None:
            ctk: dict[str, Any] = {"thinking": eff != "none"}
            if eff in ("low", "medium", "high"):
                payload["reasoning_effort"] = eff
            payload["chat_template_kwargs"] = ctk

        # explicit extra_body wins — merges on top
        if self.extra_body:
            user_ctk = self.extra_body.get("chat_template_kwargs") or {}
            merged_ctk = {**(payload.get("chat_template_kwargs") or {}), **user_ctk}
            payload.update({k: v for k, v in self.extra_body.items() if k != "chat_template_kwargs"})
            if merged_ctk:
                payload["chat_template_kwargs"] = merged_ctk

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
                # (DeepSeek-R1 / Nemotron / gpt-oss). We silently consume it —
                # the user only wants the final answer.
                _ = delta.get("reasoning_content")

                # Content channel — strip inline <think>...</think> blocks
                # before forwarding. Carry partial tokens across chunks so we
                # don't accidentally split a tag boundary.
                if (txt := delta.get("content")):
                    clean, in_inline_think, content_carry = _strip_think(
                        txt, in_inline_think, content_carry
                    )
                    if clean:
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
            yield TextDelta(content_carry)

        yield Done(finish_reason=finish_reason, usage=usage)


# ---------------------------------------------------------------------------


_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def _longest_prefix_suffix(s: str, tag: str) -> int:
    """Length of the longest non-empty suffix of `s` that is a prefix of `tag`."""
    max_k = min(len(s), len(tag) - 1)
    for k in range(max_k, 0, -1):
        if tag.startswith(s[-k:]):
            return k
    return 0


def _strip_think(chunk: str, in_think: bool, carry: str) -> tuple[str, bool, str]:
    """Strip <think>...</think> blocks from a streaming content chunk.

    Returns (visible_text, new_in_think_state, new_carry).

    `carry` holds the tail of the previous chunk that might be a partial tag —
    e.g. if a chunk ended with "<thi" we hold those 4 chars so we can detect
    "<thi" + "nk>" = open tag when the next chunk arrives.
    """
    buf = carry + chunk
    out: list[str] = []
    i = 0

    while i < len(buf):
        if in_think:
            close_idx = buf.find(_THINK_CLOSE, i)
            if close_idx < 0:
                # Still inside think and no close tag in sight. Hold only the
                # tail that could be the start of </think>; drop the rest.
                rest = buf[i:]
                k = _longest_prefix_suffix(rest, _THINK_CLOSE)
                return ("".join(out), True, rest[-k:] if k else "")
            i = close_idx + len(_THINK_CLOSE)
            in_think = False
        else:
            open_idx = buf.find(_THINK_OPEN, i)
            if open_idx < 0:
                # No more open tag in this buffer. Emit everything up to
                # whatever tail could be the start of <think>.
                rest = buf[i:]
                k = _longest_prefix_suffix(rest, _THINK_OPEN)
                emit_until = len(rest) - k
                if emit_until > 0:
                    out.append(rest[:emit_until])
                return ("".join(out), False, rest[emit_until:])
            if open_idx > i:
                out.append(buf[i:open_idx])
            i = open_idx + len(_THINK_OPEN)
            in_think = True

    return ("".join(out), in_think, "")
