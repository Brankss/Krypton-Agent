"""Shared chain-of-thought handling for providers.

Two responsibilities:

  1. Normalize the `reasoning_effort` value (none | low | medium | high)
     so every provider exposes the same `/reasoning` Telegram interface.

  2. Strip <think>...</think> blocks from streaming `content` chunks
     in a way that survives tag splits across chunk boundaries
     (DeepSeek / Nemotron / Qwen3 sometimes inline their reasoning
     directly in the content channel instead of a separate field).
"""
from __future__ import annotations

from typing import Literal

ReasoningEffort = Literal["none", "low", "medium", "high"]
VALID_EFFORTS: frozenset[str] = frozenset({"none", "low", "medium", "high"})


def normalize_effort(value: str | None) -> ReasoningEffort | None:
    """Validate and lowercase a reasoning_effort value. Raises ValueError on bad input."""
    if value is None:
        return None
    v = value.strip().lower()
    if v == "":
        return None
    if v not in VALID_EFFORTS:
        raise ValueError(
            f"reasoning_effort must be one of {sorted(VALID_EFFORTS)} or None, got {value!r}"
        )
    return v  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# <think>...</think> streaming stripper
# ---------------------------------------------------------------------------


THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def _longest_prefix_suffix(s: str, tag: str) -> int:
    """Length of the longest non-empty suffix of `s` that is a prefix of `tag`."""
    max_k = min(len(s), len(tag) - 1)
    for k in range(max_k, 0, -1):
        if tag.startswith(s[-k:]):
            return k
    return 0


def strip_think(chunk: str, in_think: bool, carry: str) -> tuple[str, bool, str]:
    """Strip <think>...</think> blocks from a streaming content chunk.

    Returns (visible_text, new_in_think_state, new_carry).

    `carry` holds the tail of the previous chunk that could be the start
    of a tag — e.g. if a chunk ended with "<thi", we hold those 4 chars
    so we can recognise "<thi" + "nk>" = open tag on the next chunk.
    """
    buf = carry + chunk
    out: list[str] = []
    i = 0

    while i < len(buf):
        if in_think:
            close_idx = buf.find(THINK_CLOSE, i)
            if close_idx < 0:
                rest = buf[i:]
                k = _longest_prefix_suffix(rest, THINK_CLOSE)
                return ("".join(out), True, rest[-k:] if k else "")
            i = close_idx + len(THINK_CLOSE)
            in_think = False
        else:
            open_idx = buf.find(THINK_OPEN, i)
            if open_idx < 0:
                rest = buf[i:]
                k = _longest_prefix_suffix(rest, THINK_OPEN)
                emit_until = len(rest) - k
                if emit_until > 0:
                    out.append(rest[:emit_until])
                return ("".join(out), False, rest[emit_until:])
            if open_idx > i:
                out.append(buf[i:open_idx])
            i = open_idx + len(THINK_OPEN)
            in_think = True

    return ("".join(out), in_think, "")
