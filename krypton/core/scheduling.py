"""Scheduling math for recurring / deferred tasks.

Pure, dependency-free, timezone-aware (stdlib zoneinfo). Kept separate from
the store and the Telegram glue so it can be unit-tested in isolation.

A schedule is one of three kinds:
  * once     — fire a single time, then disable.
  * interval — fire every N seconds.
  * daily    — fire every day at a wall-clock HH:MM in a given timezone.
"""
from __future__ import annotations

import datetime as dt

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover - zoneinfo always present on 3.11
    ZoneInfo = None  # type: ignore[assignment]

ScheduleKind = str  # "once" | "interval" | "daily"


def tzinfo(name: str) -> dt.tzinfo:
    """Resolve an IANA tz name to a tzinfo, falling back to UTC."""
    if not name or ZoneInfo is None:
        return dt.timezone.utc
    try:
        return ZoneInfo(name)
    except Exception:  # noqa: BLE001 - bad name / missing tzdata
        return dt.timezone.utc


def parse_hh_mm(s: str) -> tuple[int, int]:
    """Parse 'HH:MM' (24h) -> (hour, minute). Raises ValueError on bad input."""
    parts = (s or "").strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"time must be 'HH:MM', got {s!r}")
    h, m = int(parts[0]), int(parts[1])
    if not (0 <= h <= 23 and 0 <= m <= 59):
        raise ValueError(f"time out of range: {s!r}")
    return h, m


def next_daily(now_epoch: float, at_time: str, tz: str) -> float:
    """Epoch of the next occurrence of `at_time` (HH:MM) strictly after now."""
    h, m = parse_hh_mm(at_time)
    zone = tzinfo(tz)
    now = dt.datetime.fromtimestamp(now_epoch, zone)
    target = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if target <= now:
        target += dt.timedelta(days=1)
    return target.timestamp()


def compute_next_run(
    kind: ScheduleKind,
    now_epoch: float,
    *,
    interval_seconds: int | None = None,
    at_time: str | None = None,
    tz: str = "UTC",
) -> float | None:
    """Next epoch a recurring schedule should fire after `now_epoch`.

    Returns None for `once` (no recurrence).
    """
    if kind == "interval":
        if not interval_seconds or interval_seconds <= 0:
            raise ValueError("interval schedule needs interval_seconds > 0")
        return now_epoch + interval_seconds
    if kind == "daily":
        if not at_time:
            raise ValueError("daily schedule needs at_time")
        return next_daily(now_epoch, at_time, tz)
    if kind == "once":
        return None
    raise ValueError(f"unknown schedule kind: {kind!r}")


def initial_plan(
    *,
    now_epoch: float,
    every_minutes: int | None = None,
    every_hours: int | None = None,
    daily_at: str | None = None,
    once_in_minutes: int | None = None,
    tz: str = "UTC",
) -> tuple[ScheduleKind, int | None, str | None, float]:
    """Turn a tool-level cadence spec into (kind, interval_seconds, at_time, next_run).

    Exactly one cadence must be provided. Raises ValueError otherwise.
    """
    given = [x is not None for x in (every_minutes, every_hours, daily_at, once_in_minutes)]
    if sum(given) != 1:
        raise ValueError(
            "specify exactly one of: every_minutes, every_hours, daily_at, once_in_minutes"
        )

    if every_minutes is not None:
        if every_minutes <= 0:
            raise ValueError("every_minutes must be > 0")
        secs = every_minutes * 60
        return "interval", secs, None, now_epoch + secs
    if every_hours is not None:
        if every_hours <= 0:
            raise ValueError("every_hours must be > 0")
        secs = every_hours * 3600
        return "interval", secs, None, now_epoch + secs
    if daily_at is not None:
        parse_hh_mm(daily_at)  # validate
        return "daily", None, daily_at, next_daily(now_epoch, daily_at, tz)
    # once_in_minutes
    if once_in_minutes is None or once_in_minutes <= 0:
        raise ValueError("once_in_minutes must be > 0")
    return "once", None, None, now_epoch + once_in_minutes * 60


def format_when(epoch: float, tz: str = "UTC") -> str:
    """Human-readable local timestamp for confirmations."""
    zone = tzinfo(tz)
    return dt.datetime.fromtimestamp(epoch, zone).strftime("%Y-%m-%d %H:%M %Z")
