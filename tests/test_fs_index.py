"""Persistent FTS5 file index (instant_find backend)."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from krypton.tools._fs_index import FileIndex


def _has_fts5() -> bool:
    try:
        c = sqlite3.connect(":memory:")
        c.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
        c.close()
        return True
    except sqlite3.OperationalError:
        return False


pytestmark = pytest.mark.skipif(not _has_fts5(), reason="sqlite build lacks FTS5")


def test_build_and_glob_search(tmp_path):
    (tmp_path / "a.py").write_text("x")
    (tmp_path / "b.txt").write_text("y")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.py").write_text("z")

    idx = FileIndex(tmp_path / "idx.db")
    count, _secs = idx.build(str(tmp_path))
    assert count >= 3

    rows = idx.search("*.py", root=str(tmp_path))
    names = {Path(p).name for p, _m, _d in rows}
    assert names == {"a.py", "c.py"}


def test_fts_substring_search(tmp_path):
    (tmp_path / "kimi_config.json").write_text("{}")
    (tmp_path / "other.txt").write_text("x")
    idx = FileIndex(tmp_path / "idx.db")
    idx.build(str(tmp_path))
    rows = idx.search("kimi", root=str(tmp_path))
    assert any("kimi_config" in p for p, _m, _d in rows)


def test_kind_filter(tmp_path):
    (tmp_path / "f.py").write_text("x")
    (tmp_path / "d").mkdir()
    idx = FileIndex(tmp_path / "idx.db")
    idx.build(str(tmp_path))

    files = idx.search("*", root=str(tmp_path), kind="file")
    dirs = idx.search("*", root=str(tmp_path), kind="dir")
    assert files and all(is_dir == 0 for _p, _m, is_dir in files)
    assert dirs and all(is_dir == 1 for _p, _m, is_dir in dirs)


def test_prune_dirs_skipped(tmp_path):
    (tmp_path / "keep.py").write_text("x")
    junk = tmp_path / "node_modules"
    junk.mkdir()
    (junk / "lib.py").write_text("x")
    idx = FileIndex(tmp_path / "idx.db")
    idx.build(str(tmp_path))
    rows = idx.search("*.py", root=str(tmp_path))
    paths = [p for p, _m, _d in rows]
    assert any("keep.py" in p for p in paths)
    assert not any("node_modules" in p for p in paths)


def test_root_age_none_before_build(tmp_path):
    idx = FileIndex(tmp_path / "idx.db")
    assert idx.root_age(str(tmp_path)) is None
    idx.build(str(tmp_path))
    age = idx.root_age(str(tmp_path))
    assert age is not None and age >= 0
