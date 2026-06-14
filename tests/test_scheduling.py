"""Pure scheduling math for recurring/deferred tasks."""
from __future__ import annotations

import datetime as dt
import time

import pytest

from krypton.core import scheduling as s


def test_parse_hh_mm_ok():
    assert s.parse_hh_mm("08:30") == (8, 30)
    assert s.parse_hh_mm("23:59") == (23, 59)
    assert s.parse_hh_mm("00:00") == (0, 0)


@pytest.mark.parametrize("bad", ["8", "24:00", "12:60", "aa:bb", "", "1:2:3"])
def test_parse_hh_mm_bad(bad):
    with pytest.raises(ValueError):
        s.parse_hh_mm(bad)


def test_interval_next():
    assert s.compute_next_run("interval", 1000.0, interval_seconds=60) == 1060.0
    with pytest.raises(ValueError):
        s.compute_next_run("interval", 1000.0, interval_seconds=0)


def test_daily_next_is_future_at_right_time():
    now = time.time()
    nxt = s.compute_next_run("daily", now, at_time="03:00", tz="UTC")
    assert nxt > now
    assert nxt - now <= 24 * 3600 + 1
    d = dt.datetime.fromtimestamp(nxt, dt.timezone.utc)
    assert (d.hour, d.minute) == (3, 0)


def test_once_has_no_recurrence():
    assert s.compute_next_run("once", 1000.0) is None


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        s.compute_next_run("weekly", 1.0)


def test_initial_plan_needs_exactly_one_cadence():
    with pytest.raises(ValueError):
        s.initial_plan(now_epoch=0)
    with pytest.raises(ValueError):
        s.initial_plan(now_epoch=0, every_minutes=5, every_hours=1)


def test_initial_plan_every_minutes():
    kind, iv, at, nxt = s.initial_plan(now_epoch=0, every_minutes=10)
    assert (kind, iv, at, nxt) == ("interval", 600, None, 600)


def test_initial_plan_every_hours():
    kind, iv, at, nxt = s.initial_plan(now_epoch=0, every_hours=2)
    assert (kind, iv, at, nxt) == ("interval", 7200, None, 7200)


def test_initial_plan_daily():
    now = time.time()
    kind, iv, at, nxt = s.initial_plan(now_epoch=now, daily_at="09:15", tz="UTC")
    assert kind == "daily" and iv is None and at == "09:15" and nxt > now


def test_initial_plan_once():
    kind, iv, at, nxt = s.initial_plan(now_epoch=100.0, once_in_minutes=30)
    assert (kind, iv, at, nxt) == ("once", None, None, 100.0 + 1800)


def test_initial_plan_rejects_nonpositive():
    with pytest.raises(ValueError):
        s.initial_plan(now_epoch=0, every_minutes=0)


def test_tzinfo_falls_back_to_utc_on_garbage():
    assert s.tzinfo("Not/AZone") is dt.timezone.utc


def test_format_when():
    assert "1970-01-01" in s.format_when(0.0, "UTC")
