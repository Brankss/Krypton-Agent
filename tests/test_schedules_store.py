"""MemoryStore: schedules table (CRUD + due-query + scoping)."""
from __future__ import annotations

from krypton.memory.store import MemoryStore


async def test_add_and_list_scoped(tmp_path):
    s = MemoryStore(tmp_path / "m.db")
    try:
        sc = await s.add_schedule(
            chat_id=1, title="t", prompt="do x", kind="interval",
            next_run=100.0, interval_seconds=60,
        )
        assert sc.id > 0
        rows = await s.list_schedules(chat_id=1)
        assert len(rows) == 1 and rows[0].title == "t" and rows[0].interval_seconds == 60
        assert await s.list_schedules(chat_id=999) == []
    finally:
        await s.aclose()


async def test_due_schedules_only_past_and_enabled(tmp_path):
    s = MemoryStore(tmp_path / "m.db")
    try:
        await s.add_schedule(chat_id=1, title="past", prompt="p", kind="interval",
                             next_run=10.0, interval_seconds=60)
        await s.add_schedule(chat_id=1, title="future", prompt="p", kind="interval",
                             next_run=1e12, interval_seconds=60)
        due = await s.due_schedules(now=1000.0)
        assert {d.title for d in due} == {"past"}
    finally:
        await s.aclose()


async def test_update_after_run(tmp_path):
    s = MemoryStore(tmp_path / "m.db")
    try:
        sc = await s.add_schedule(chat_id=1, title="t", prompt="p", kind="interval",
                                  next_run=10.0, interval_seconds=60)
        await s.update_after_run(sc.id, next_run=5000.0, last_run=1000.0)
        row = (await s.list_schedules(chat_id=1))[0]
        assert row.next_run == 5000.0 and row.last_run == 1000.0
    finally:
        await s.aclose()


async def test_disable_hides_from_due_and_default_list(tmp_path):
    s = MemoryStore(tmp_path / "m.db")
    try:
        sc = await s.add_schedule(chat_id=1, title="t", prompt="p", kind="once", next_run=10.0)
        await s.set_schedule_enabled(sc.id, False, last_run=1.0)
        assert await s.due_schedules(now=1000.0) == []
        assert await s.list_schedules(chat_id=1) == []                      # enabled-only by default
        assert len(await s.list_schedules(chat_id=1, include_disabled=True)) == 1
    finally:
        await s.aclose()


async def test_delete_is_chat_scoped(tmp_path):
    s = MemoryStore(tmp_path / "m.db")
    try:
        sc = await s.add_schedule(chat_id=1, title="t", prompt="p", kind="once", next_run=10.0)
        assert await s.delete_schedule(sc.id, chat_id=2) is False   # wrong chat: no-op
        assert await s.delete_schedule(sc.id, chat_id=1) is True
        assert await s.list_schedules(chat_id=1, include_disabled=True) == []
    finally:
        await s.aclose()
