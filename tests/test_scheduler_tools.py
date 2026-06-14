"""Agent-facing scheduler tools (schedule_task / list_schedules / cancel_schedule)."""
from __future__ import annotations

from krypton.memory.store import MemoryStore
from krypton.tools.scheduler import CancelScheduleTool, ListSchedulesTool, ScheduleTaskTool


async def test_schedule_task_interval(tmp_path):
    s = MemoryStore(tmp_path / "m.db")
    try:
        res = await ScheduleTaskTool(s, chat_id=42).run(
            title="ping", prompt="say hi", every_minutes=15
        )
        assert res.ok and "scheduled #" in res.content
        rows = await s.list_schedules(chat_id=42)
        assert len(rows) == 1 and rows[0].kind == "interval" and rows[0].interval_seconds == 900
        assert rows[0].chat_id == 42
    finally:
        await s.aclose()


async def test_schedule_task_requires_exactly_one_cadence(tmp_path):
    s = MemoryStore(tmp_path / "m.db")
    try:
        t = ScheduleTaskTool(s, chat_id=1)
        none = await t.run(title="x", prompt="y")
        assert not none.ok and "exactly one" in none.content
        both = await t.run(title="x", prompt="y", every_minutes=5, daily_at="08:00")
        assert not both.ok
        assert await s.list_schedules(chat_id=1) == []
    finally:
        await s.aclose()


async def test_schedule_task_daily_and_once(tmp_path):
    s = MemoryStore(tmp_path / "m.db")
    try:
        t = ScheduleTaskTool(s, chat_id=1)
        assert (await t.run(title="rep", prompt="daily report", daily_at="07:00")).ok
        assert (await t.run(title="rem", prompt="remind me", once_in_minutes=90)).ok
        kinds = {r.kind for r in await s.list_schedules(chat_id=1)}
        assert kinds == {"daily", "once"}
    finally:
        await s.aclose()


async def test_list_and_cancel_are_chat_scoped(tmp_path):
    s = MemoryStore(tmp_path / "m.db")
    try:
        await ScheduleTaskTool(s, chat_id=1).run(title="x", prompt="y", every_minutes=5)
        sid = (await s.list_schedules(chat_id=1))[0].id

        # another chat cannot see or cancel it
        assert "no scheduled tasks" in (await ListSchedulesTool(s, chat_id=2).run()).content
        assert not (await CancelScheduleTool(s, chat_id=2).run(id=sid)).ok

        listed = await ListSchedulesTool(s, chat_id=1).run()
        assert f"#{sid}" in listed.content
        assert (await CancelScheduleTool(s, chat_id=1).run(id=sid)).ok
        assert "no scheduled tasks" in (await ListSchedulesTool(s, chat_id=1).run()).content
    finally:
        await s.aclose()
