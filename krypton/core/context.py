"""Conversation context manager.

Not 'dumb memory'. Four principles:

1. **Deduplication first.** If the same tool result, error, or assistant
   message reappears N times, we collapse the older copies into a stub
   and tag the latest with "this occurred N× in this conversation".
   That way 50 identical "permission denied" errors take the space of 1.

2. **Tool results are first-class but compressible.** Big results
   (file dumps, grep output) get truncated head+tail after they've
   served their purpose.

3. **Hard token budget.** We compact aggressively once we cross
   `target_budget`. Compaction never drops user messages or the most
   recent 6 turns.

4. **Pinned messages.** Anything tagged `pinned=True` (e.g. the user's
   original goal) survives every compaction.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from krypton.core.tokens import count_tokens
from krypton.providers.base import Message, ToolCall


@dataclass(slots=True)
class _Entry:
    msg: Message
    tokens: int
    pinned: bool = False
    fp: str | None = None  # cached fingerprint for dedup; invalidated on content change

    def invalidate(self) -> None:
        self.fp = None


class ConversationContext:
    def __init__(self, target_budget: int = 24000) -> None:
        self.target_budget = target_budget
        self._entries: list[_Entry] = []
        self._system: Message | None = None

    # ------------------------------------------------------------------
    def set_system(self, content: str) -> None:
        self._system = Message(role="system", content=content)

    def add(self, msg: Message, *, pinned: bool = False) -> None:
        self._entries.append(_Entry(msg=msg, tokens=_estimate(msg), pinned=pinned))

    def add_user(self, text: str, *, pinned: bool = True) -> None:
        self.add(Message(role="user", content=text), pinned=pinned)

    def add_assistant(self, msg: Message) -> None:
        self.add(msg)

    def add_tool_result(self, *, tool_call_id: str, name: str, content: str) -> None:
        self.add(Message(role="tool", content=content, tool_call_id=tool_call_id, name=name))

    # ------------------------------------------------------------------
    def messages(self) -> list[Message]:
        out: list[Message] = []
        if self._system:
            out.append(self._system)
        out.extend(e.msg for e in self._entries)
        return out

    @property
    def token_count(self) -> int:
        total = count_tokens(self._system.content) if self._system else 0
        return total + sum(e.tokens for e in self._entries)

    @property
    def entry_count(self) -> int:
        """Number of non-system entries (public accessor for /status etc.)."""
        return len(self._entries)

    # ------------------------------------------------------------------
    def compact(self) -> int:
        """Reduce context size in stages.

        1) Deduplicate exact-repeat tool results / assistant turns.
        2) If still over budget: head/tail-truncate large old tool outputs.
        3) If still over budget: drop assistant scratch turns from the middle.
        4) Always: repair call/result pairing so truncation never leaves an
           orphaned tool message (or a dangling assistant tool_call) — that
           is exactly what makes OpenAI-format providers reject the history.

        Returns total tokens freed across all stages.
        """
        freed = self._deduplicate()

        if self.token_count > self.target_budget:
            cutoff = max(0, len(self._entries) - 6)
            for i in range(cutoff):
                e = self._entries[i]
                if e.pinned or e.msg.role != "tool":
                    continue
                if e.tokens < 200:
                    continue
                old = e.msg.content
                e.msg.content = _head_tail(old, head=600, tail=400)
                new_t = _estimate(e.msg)
                freed += e.tokens - new_t
                e.tokens = new_t
                e.invalidate()
                if self.token_count <= self.target_budget:
                    break

        if self.token_count > self.target_budget:
            keep: list[_Entry] = []
            tail = self._entries[-6:]
            for e in self._entries[:-6]:
                if e.pinned or e.msg.role == "user":
                    keep.append(e)
            self._entries = keep + tail

        # Final safety net: stages 2-3 can drop an assistant turn while keeping
        # its tool result (or vice versa). Repair the pairing unconditionally.
        freed += self._repair_pairs()
        return freed

    # ------------------------------------------------------------------
    def _deduplicate(self) -> int:
        """Drop duplicate tool/assistant entries; keep only the latest copy
        tagged with `[repeated N×]`. Pair-aware so we never leave a tool
        message orphaned (its assistant tool_call gone) or vice versa.

        Rules:
          - User messages are NEVER deduplicated (each is a fresh intent).
          - Pinned entries are skipped.
          - Very short entries (<80 chars, no tool_calls) are skipped.
          - The LATEST occurrence stays in place with a count tag.
          - All EARLIER occurrences are removed from the list entirely.
          - Post-pass: any tool message whose tool_call_id no longer points
            at a preceding assistant is dropped, and any assistant tool_call
            without a following tool response is stripped.
        """
        if len(self._entries) < 3:
            return 0

        # Bucket entries by fingerprint (cached per entry; recomputed only when invalidated)
        groups: dict[str, list[int]] = {}
        for i, e in enumerate(self._entries):
            if e.pinned:
                continue
            if e.msg.role == "user":
                continue
            if len(e.msg.content or "") < 80 and not e.msg.tool_calls:
                continue
            if e.fp is None:
                e.fp = _fingerprint(e.msg)
            groups.setdefault(e.fp, []).append(i)

        to_drop: set[int] = set()
        for fp, indices in groups.items():
            if len(indices) < 2:
                continue
            canonical = self._entries[indices[-1]]
            count = len(indices)
            if not (canonical.msg.content or "").startswith("[repeated"):
                tag = f"[repeated {count}× in this conversation; older copies dropped]\n"
                canonical.msg.content = tag + (canonical.msg.content or "")
                canonical.tokens = _estimate(canonical.msg)
                canonical.invalidate()
            to_drop.update(indices[:-1])

        if not to_drop:
            return 0

        freed = sum(self._entries[i].tokens for i in to_drop)
        self._entries = [e for i, e in enumerate(self._entries) if i not in to_drop]

        # Dropping duplicates can sever a call/result pair; repair it.
        freed += self._repair_pairs()
        return freed

    # ------------------------------------------------------------------
    def _repair_pairs(self) -> int:
        """Keep assistant tool_calls and tool responses paired.

        Two passes, idempotent and safe to run after any structural mutation
        (dedup, head/tail truncation, scratch drop):

          1. Drop any tool message whose `tool_call_id` no longer points at a
             preceding assistant tool_call.
          2. Strip any assistant tool_call that has no following tool response.

        Returns tokens freed. This is the invariant OpenAI-format providers
        (OpenRouter / NVIDIA) enforce: an unpaired tool message or a dangling
        assistant tool_call is a hard 400.
        """
        freed = 0

        # --- pass 1: drop orphan tool messages ----------------------------
        valid_call_ids: set[str] = set()
        for e in self._entries:
            if e.msg.role == "assistant":
                for tc in e.msg.tool_calls:
                    valid_call_ids.add(tc.id)
        orphans = {
            i for i, e in enumerate(self._entries)
            if e.msg.role == "tool"
            and e.msg.tool_call_id
            and e.msg.tool_call_id not in valid_call_ids
        }
        if orphans:
            freed += sum(self._entries[i].tokens for i in orphans)
            self._entries = [e for i, e in enumerate(self._entries) if i not in orphans]

        # --- pass 2: strip assistant tool_calls without responses ---------
        answered: set[str] = set()
        for e in self._entries:
            if e.msg.role == "tool" and e.msg.tool_call_id:
                answered.add(e.msg.tool_call_id)
        for e in self._entries:
            if e.msg.role == "assistant" and e.msg.tool_calls:
                kept = [tc for tc in e.msg.tool_calls if tc.id in answered]
                if len(kept) != len(e.msg.tool_calls):
                    before = e.tokens
                    e.msg.tool_calls = kept
                    e.tokens = _estimate(e.msg)
                    e.invalidate()
                    freed += before - e.tokens

        return freed

    # ------------------------------------------------------------------
    def drop_trailing_unanswered_tool_calls(self) -> None:
        """Strip tool_calls from the final assistant message if they were never
        answered (e.g. the iteration cap was hit mid-dispatch), so the next
        turn doesn't start from a structurally broken history."""
        if not self._entries:
            return
        last = self._entries[-1]
        if last.msg.role == "assistant" and last.msg.tool_calls:
            last.msg.tool_calls = []
            last.tokens = _estimate(last.msg)
            last.invalidate()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._entries.clear()

    # ------------------------------------------------------------------
    # Serialization for cross-restart persistence.
    # System prompt is intentionally NOT saved: it's rebuilt every turn
    # from the live environment + memory.
    # ------------------------------------------------------------------
    def serialize(self) -> str:
        out = []
        for e in self._entries:
            m = e.msg
            out.append(
                {
                    "role": m.role,
                    "content": m.content,
                    "tool_calls": [
                        {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                        for tc in m.tool_calls
                    ],
                    "tool_call_id": m.tool_call_id,
                    "name": m.name,
                    "tokens": e.tokens,
                    "pinned": e.pinned,
                }
            )
        return json.dumps(out, ensure_ascii=False)

    def restore(self, payload: str) -> int:
        data = json.loads(payload) if payload else []
        self._entries.clear()
        for d in data:
            tcs = [
                ToolCall(id=t["id"], name=t["name"], arguments=t.get("arguments") or {})
                for t in (d.get("tool_calls") or [])
            ]
            msg = Message(
                role=d.get("role") or "user",
                content=d.get("content") or "",
                tool_calls=tcs,
                tool_call_id=d.get("tool_call_id"),
                name=d.get("name"),
            )
            self._entries.append(
                _Entry(
                    msg=msg,
                    tokens=int(d.get("tokens") or _estimate(msg)),
                    pinned=bool(d.get("pinned", False)),
                )
            )
        return len(self._entries)


def _estimate(msg: Message) -> int:
    n = count_tokens(msg.content or "")
    for tc in msg.tool_calls:
        n += count_tokens(tc.name) + count_tokens(str(tc.arguments))
    return n


def _head_tail(text: str, *, head: int, tail: int) -> str:
    if len(text) <= head + tail + 64:
        return text
    return (
        text[:head]
        + f"\n\n... [{len(text) - head - tail} chars elided to save context] ...\n\n"
        + text[-tail:]
    )


def _normalize(text: str) -> str:
    """Lowercase + collapse whitespace — strips superficial differences
    (extra spaces, trailing newlines) before hashing."""
    return " ".join(text.split()).lower()


def _content_hash(text: str) -> str:
    return hashlib.sha1(_normalize(text).encode("utf-8", "replace")).hexdigest()[:16]


def _fingerprint(msg: Message) -> str:
    """Fingerprint a message for deduplication.

    Includes:
      - role (so a tool result and an assistant echo never collide),
      - tool name when role='tool',
      - hashed tool-call signatures (name + sorted JSON args) for assistant turns,
      - hash of the normalized text body.
    """
    parts: list[str] = [msg.role]
    if msg.role == "tool" and msg.name:
        parts.append(f"t={msg.name}")
    if msg.tool_calls:
        for tc in msg.tool_calls:
            try:
                args = json.dumps(tc.arguments, sort_keys=True, ensure_ascii=True)
            except (TypeError, ValueError):
                args = repr(tc.arguments)
            parts.append(f"c={tc.name}:{_content_hash(args)}")
    body = msg.content or ""
    if body:
        parts.append(f"b={_content_hash(body)}")
    return "|".join(parts)
