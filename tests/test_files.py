"""File tools: read/write/edit + the delete_path protected-path guard.

Autonomy is intentional, so the guard is the only thing standing between a
hallucinated `delete_path('.', recursive=True)` and a wiped workdir. Tests
monkeypatch settings.workdir to a temp dir so a guard regression can never
endanger the real repository.
"""
from __future__ import annotations

from pathlib import Path

from krypton.config import settings
from krypton.tools.files import (
    DeletePathTool,
    EditFileTool,
    ReadFileTool,
    WriteFileTool,
    _is_protected,
)


async def test_write_then_read_roundtrip(tmp_path):
    f = tmp_path / "a.txt"
    w = await WriteFileTool().run(str(f), "hello\nworld\n")
    assert w.ok
    r = await ReadFileTool().run(str(f))
    assert r.ok and "hello" in r.content and "world" in r.content


async def test_edit_requires_unique_match(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("x\nx\n")
    res = await EditFileTool().run(str(f), "x", "y")
    assert not res.ok and "unique" in res.content
    res2 = await EditFileTool().run(str(f), "x", "y", replace_all=True)
    assert res2.ok
    assert f.read_text() == "y\ny\n"


async def test_edit_missing_string(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("abc")
    res = await EditFileTool().run(str(f), "zzz", "y")
    assert not res.ok and "not found" in res.content


def test_is_protected_true_for_roots():
    assert _is_protected(settings.workdir) is True
    assert _is_protected(settings.data_dir) is True
    assert _is_protected(Path.home()) is True
    assert _is_protected(Path(settings.workdir.anchor)) is True  # "/" or "C:\\"


def test_is_protected_false_for_normal_path():
    assert _is_protected(Path("/some/unrelated/deep/child.txt")) is False


def test_is_protected_true_for_ancestor_of_workdir():
    # An ancestor of the workdir would take the workdir down with it.
    assert _is_protected(settings.workdir.parent) is True


async def test_delete_refuses_workdir_root(tmp_path, monkeypatch):
    # Point the agent's "home" at a temp dir; "." then resolves to it and must
    # be refused. Even if the guard regressed, only tmp_path is ever at risk.
    monkeypatch.setattr(settings, "workdir", tmp_path)
    (tmp_path / "keep.txt").write_text("data")
    res = await DeletePathTool().run(".", recursive=True)
    assert not res.ok and "protected" in res.content
    assert (tmp_path / "keep.txt").exists()  # nothing was deleted


async def test_delete_removes_child(tmp_path):
    f = tmp_path / "gone.txt"
    f.write_text("bye")
    res = await DeletePathTool().run(str(f))
    assert res.ok and not f.exists()


async def test_delete_dir_requires_recursive(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    (d / "f.txt").write_text("x")
    res = await DeletePathTool().run(str(d))  # recursive defaults False
    assert not res.ok
    assert d.exists()
    res2 = await DeletePathTool().run(str(d), recursive=True)
    assert res2.ok and not d.exists()
