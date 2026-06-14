"""Tools that let the agent run things on a schedule.

These make Krypton proactive: it can register one-off or recurring tasks that
fire by themselves (even while the user is away) and message the result back.

Bound per chat (like send_file_to_user): the chat_id is captured at
construction so a scheduled task always reports to the right conversation.
"""
from __future__ import annotations

import time
from typing import Any

from krypton.config import settings
from krypton.core.scheduling import format_when, initial_plan
from krypton.memory.store import MemoryStore
from krypton.tools.base import BaseTool, ToolResult


class ScheduleTaskTool(BaseTool):
    name = "schedule_task"
    description = (
        "Register a task to run automatically later — once or recurring. Use this "
        "whenever the user wants something on a cadence or at a time ('ogni mattina', "
        "' tutti i lunedì', 'tra un'ora', 'ricordami', ' every day', 'each week'). "
        "The task runs by itself and its result is sent to this chat. Give exactly ONE "
        "cadence. The `prompt` is the instruction future-you will execute, so write it "
        "self-contained (don't rely on the current conversation)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Short label, e.g. 'Daily AI trends report'."},
            "prompt": {
                "type": "string",
                "description": "Self-contained instruction to execute when it fires.",
            },
            "every_minutes": {"type": "integer", "description": "Recur every N minutes."},
            "every_hours": {"type": "integer", "description": "Recur every N hours."},
            "daily_at": {"type": "string", "description": "Recur daily at 'HH:MM' (24h, local tz)."},
            "once_in_minutes": {"type": "integer", "description": "Run a single time, N minutes from now."},
        },
        "required": ["title", "prompt"],
    }
    timeout_s = 5.0

    def __init__(self, store: MemoryStore, chat_id: int) -> None:
        self._store = store
        self._chat_id = chat_id

    async def run(
        self,
        title: str,
        prompt: str,
        every_minutes: int | None = None,
        every_hours: int | None = None,
        daily_at: str | None = None,
        once_in_minutes: int | None = None,
    ) -> ToolResult:
        try:
            kind, interval_seconds, at_time, next_run = initial_plan(
                now_epoch=time.time(),
                every_minutes=every_minutes,
                every_hours=every_hours,
                daily_at=daily_at,
                once_in_minutes=once_in_minutes,
                tz=settings.timezone,
            )
        except ValueError as e:
            return ToolResult.failure(str(e))

        sched = await self._store.add_schedule(
            chat_id=self._chat_id,
            title=title.strip(),
            prompt=prompt.strip(),
            kind=kind,
            next_run=next_run,
            interval_seconds=interval_seconds,
            at_time=at_time,
        )
        cadence = (
            f"every {interval_seconds // 60} min" if kind == "interval" and interval_seconds and interval_seconds < 3600
            else f"every {interval_seconds // 3600} h" if kind == "interval" and interval_seconds
            else f"daily at {at_time}" if kind == "daily"
            else "once"
        )
        return ToolResult.success(
            f"scheduled #{sched.id} '{sched.title}' ({cadence}); next run {format_when(next_run, settings.timezone)}"
        )


class ListSchedulesTool(BaseTool):
    name = "list_schedules"
    description = "List the active scheduled tasks for this chat (id, title, cadence, next run)."
    parameters = {"type": "object", "properties": {}}
    timeout_s = 5.0

    def __init__(self, store: MemoryStore, chat_id: int) -> None:
        self._store = store
        self._chat_id = chat_id

    async def run(self) -> ToolResult:
        rows = await self._store.list_schedules(chat_id=self._chat_id)
        if not rows:
            return ToolResult.success("no scheduled tasks")
        lines = []
        for s in rows:
            if s.kind == "interval" and s.interval_seconds:
                cad = f"every {s.interval_seconds // 60}min"
            elif s.kind == "daily":
                cad = f"daily at {s.at_time}"
            else:
                cad = "once"
            lines.append(f"#{s.id} {s.title} — {cad}, next {format_when(s.next_run, settings.timezone)}")
        return ToolResult.success("\n".join(lines))


class CancelScheduleTool(BaseTool):
    name = "cancel_schedule"
    description = "Cancel (delete) a scheduled task by its id. Use list_schedules first to find the id."
    parameters = {
        "type": "object",
        "properties": {"id": {"type": "integer"}},
        "required": ["id"],
    }
    timeout_s = 5.0

    def __init__(self, store: MemoryStore, chat_id: int) -> None:
        self._store = store
        self._chat_id = chat_id

    async def run(self, id: int) -> ToolResult:  # noqa: A002 - LLM-facing name
        ok = await self._store.delete_schedule(id, chat_id=self._chat_id)
        return ToolResult(ok=ok, content=f"cancelled #{id}" if ok else f"no schedule #{id} in this chat")


def tools(store: MemoryStore, chat_id: int) -> list[Any]:
    return [
        ScheduleTaskTool(store, chat_id),
        ListSchedulesTool(store, chat_id),
        CancelScheduleTool(store, chat_id),
    ]
