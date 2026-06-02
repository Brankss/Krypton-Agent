"""Conversation context: dedup, pair-safety, truncation, persistence.

The headline regression target is the call/result pairing invariant: an
OpenAI-format provider (OpenRouter / NVIDIA) hard-400s on a tool message with
no preceding assistant tool_call, or an assistant tool_call with no following
tool response. Compaction must NEVER leave the history in that state.
"""
from __future__ import annotations

from krypton.core.context import ConversationContext
from krypton.providers.base import Message, ToolCall


def _assistant_call(call_id: str, name: str = "grep", args=None, text: str = "") -> Message:
    return Message(role="assistant", content=text,
                   tool_calls=[ToolCall(id=call_id, name=name, arguments=args or {})])


def _pairing_is_valid(msgs: list[Message]) -> bool:
    call_ids = {tc.id for m in msgs if m.role == "assistant" for tc in m.tool_calls}
    tool_ids = {m.tool_call_id for m in msgs if m.role == "tool" and m.tool_call_id}
    # every tool message answers a real call, and every call is answered
    if not tool_ids <= call_ids:
        return False
    return call_ids <= tool_ids


def test_entry_count():
    c = ConversationContext()
    assert c.entry_count == 0
    c.add_user("hi")
    c.add_user("there")
    assert c.entry_count == 2


def test_repair_pairs_drops_orphan_tool():
    c = ConversationContext()
    c.add_user("goal")
    c.add_tool_result(tool_call_id="ghost", name="grep", content="x")  # no matching call
    c._repair_pairs()
    assert all(m.role != "tool" for m in c.messages())


def test_repair_pairs_strips_dangling_assistant_call():
    c = ConversationContext()
    c.add_user("goal")
    c.add_assistant(_assistant_call("c1"))  # call never answered
    c._repair_pairs()
    a = [m for m in c.messages() if m.role == "assistant"][0]
    assert a.tool_calls == []


def test_compact_preserves_pairing_invariant_under_heavy_truncation():
    # Tiny budget forces stage-2/3 truncation; the final repair pass must keep
    # the call/result pairing valid no matter where the cut lands.
    c = ConversationContext(target_budget=1)
    c.add_user("the original goal", pinned=True)
    for i in range(12):
        cid = f"call_{i}"
        c.add_assistant(_assistant_call(cid, text="reasoning " * 10))
        c.add_tool_result(tool_call_id=cid, name="grep", content="Z" * 500)
    c.compact()
    assert _pairing_is_valid(c.messages())


def test_compact_preserves_pairing_with_multi_call_turns():
    c = ConversationContext(target_budget=1)
    c.add_user("goal", pinned=True)
    for i in range(8):
        a, b = f"c{i}a", f"c{i}b"
        c.add_assistant(Message(role="assistant", content="x" * 60, tool_calls=[
            ToolCall(id=a, name="read_file", arguments={}),
            ToolCall(id=b, name="grep", arguments={}),
        ]))
        c.add_tool_result(tool_call_id=a, name="read_file", content="A" * 300)
        c.add_tool_result(tool_call_id=b, name="grep", content="B" * 300)
    c.compact()
    assert _pairing_is_valid(c.messages())


def test_dedup_collapses_identical_tool_results():
    c = ConversationContext(target_budget=10_000_000)  # high: isolate dedup from truncation
    c.add_user("goal")
    for i in range(5):
        cid = f"c{i}"
        c.add_assistant(_assistant_call(cid))
        c.add_tool_result(tool_call_id=cid, name="grep", content="E" * 120)  # >80 chars => eligible
    c.compact()
    tool_msgs = [m for m in c.messages() if m.role == "tool"]
    assert len(tool_msgs) == 1
    assert "[repeated 5×" in tool_msgs[0].content
    assert _pairing_is_valid(c.messages())


def test_dedup_never_touches_user_messages():
    c = ConversationContext(target_budget=10_000_000)
    for _ in range(4):
        c.add_user("do the exact same thing again please now")
    c.compact()
    users = [m for m in c.messages() if m.role == "user"]
    assert len(users) == 4  # each user turn is a fresh intent, never deduped


def test_drop_trailing_unanswered_tool_calls():
    c = ConversationContext()
    c.add_user("x")
    c.add_assistant(_assistant_call("c1"))
    c.drop_trailing_unanswered_tool_calls()
    last = c.messages()[-1]
    assert last.role == "assistant" and last.tool_calls == []


def test_serialize_restore_roundtrip():
    c = ConversationContext()
    c.add_user("hello", pinned=True)
    c.add_assistant(_assistant_call("c1", text="working"))
    c.add_tool_result(tool_call_id="c1", name="grep", content="found it")
    payload = c.serialize()

    c2 = ConversationContext()
    n = c2.restore(payload)
    assert n == 3
    m = c2.messages()
    assert m[0].role == "user" and m[0].content == "hello"
    assert m[1].role == "assistant" and m[1].tool_calls[0].id == "c1"
    assert m[2].role == "tool" and m[2].tool_call_id == "c1"


def test_pinned_and_first_user_survive_truncation():
    c = ConversationContext(target_budget=1)
    c.add_user("FIRST GOAL", pinned=True)
    for i in range(20):
        cid = f"c{i}"
        c.add_assistant(_assistant_call(cid, text="x" * 40))
        c.add_tool_result(tool_call_id=cid, name="grep", content="y" * 40)
    c.compact()
    users = [m for m in c.messages() if m.role == "user"]
    assert any(m.content == "FIRST GOAL" for m in users)
