"""File CRUD tools.

  * read_file        — slice with offset/limit, never blows context
  * write_file       — atomic, creates parents
  * edit_file        — exact string replacement with optional replace_all
  * append_file      — append text to an existing or new file
  * delete_path      — delete file or recursive dir
  * stat_path        — quick metadata probe
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import aiofiles

from krypton.config import settings
from krypton.tools.base import BaseTool, ToolResult

_DEFAULT_READ_LIMIT = 2000  # lines
_MAX_FILE_READ_BYTES = 4 * 1024 * 1024


def _resolve(path: str) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = settings.workdir / p
    return p


class ReadFileTool(BaseTool):
    name = "read_file"
    description = (
        "Read a text file. Returns content with 1-indexed line numbers (cat -n style). "
        "Use `offset` + `limit` for large files; default cap is 2000 lines."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "offset": {"type": "integer", "default": 0, "description": "0-indexed start line."},
            "limit": {"type": "integer", "default": _DEFAULT_READ_LIMIT},
        },
        "required": ["path"],
    }
    timeout_s = 15.0

    async def run(self, path: str, offset: int = 0, limit: int = _DEFAULT_READ_LIMIT) -> ToolResult:
        target = _resolve(path)
        if not target.exists():
            return ToolResult.failure(f"file not found: {target}")
        if not target.is_file():
            return ToolResult.failure(f"not a regular file: {target}")
        try:
            st = target.stat()
        except OSError as e:
            return ToolResult.failure(f"stat failed: {e}")
        if st.st_size > _MAX_FILE_READ_BYTES:
            return ToolResult.failure(
                f"file is {st.st_size} bytes (> {_MAX_FILE_READ_BYTES}). "
                "Use grep or a python script to slice it."
            )

        # Stream lines so we don't load the whole file when slicing into the middle.
        import asyncio

        def _slice() -> tuple[list[str], int]:
            sliced: list[str] = []
            total = 0
            with open(target, "r", encoding="utf-8", errors="replace") as f:
                for i, line in enumerate(f):
                    total = i + 1
                    if i < offset:
                        continue
                    if len(sliced) < limit:
                        sliced.append(line.rstrip("\n"))
                    # keep counting `total` even after we have enough lines
            return sliced, total

        sliced, total = await asyncio.to_thread(_slice)
        end = min(total, offset + len(sliced))
        width = max(4, len(str(end)))
        body = "\n".join(f"{i + 1:>{width}}\t{ln}" for i, ln in enumerate(sliced, start=offset))
        header = f"{target}  ({total} lines, showing {offset + 1}-{end})"
        return ToolResult.success(f"{header}\n{body}", meta={"total_lines": total})


class WriteFileTool(BaseTool):
    name = "write_file"
    description = (
        "Write (or overwrite) a UTF-8 text file atomically (temp file + rename). "
        "Creates parent directories. Use `edit_file` for surgical changes."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    }
    timeout_s = 20.0

    async def run(self, path: str, content: str) -> ToolResult:
        target = _resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        def _write_atomic() -> int:
            fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".krypton_", suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
                    n = f.write(content)
                os.replace(tmp, target)
                return n
            except Exception:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise

        import asyncio
        n = await asyncio.to_thread(_write_atomic)
        return ToolResult.success(f"wrote {n} chars -> {target}", meta={"bytes": n})


class EditFileTool(BaseTool):
    name = "edit_file"
    description = (
        "Replace an exact substring in a file. Fails if old_string isn't unique, "
        "unless `replace_all=true`. Preserve indentation exactly when matching."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old_string": {"type": "string"},
            "new_string": {"type": "string"},
            "replace_all": {"type": "boolean", "default": False},
        },
        "required": ["path", "old_string", "new_string"],
    }
    timeout_s = 20.0

    async def run(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> ToolResult:
        target = _resolve(path)
        if not target.is_file():
            return ToolResult.failure(f"file not found: {target}")
        if old_string == new_string:
            return ToolResult.failure("old_string and new_string are identical")

        async with aiofiles.open(target, "r", encoding="utf-8", errors="replace") as f:
            text = await f.read()

        occurrences = text.count(old_string)
        if occurrences == 0:
            return ToolResult.failure("old_string not found in file")
        if occurrences > 1 and not replace_all:
            return ToolResult.failure(
                f"old_string is not unique ({occurrences} matches). "
                "Give a longer context or set replace_all=true."
            )

        new_text = text.replace(old_string, new_string) if replace_all else text.replace(old_string, new_string, 1)
        await _atomic_write_text(target, new_text)
        return ToolResult.success(
            f"edited {target}: replaced {occurrences if replace_all else 1} occurrence(s)",
            meta={"replacements": occurrences if replace_all else 1},
        )


class AppendFileTool(BaseTool):
    name = "append_file"
    description = "Append text to a file (creates it if missing)."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    }
    timeout_s = 15.0

    async def run(self, path: str, content: str) -> ToolResult:
        target = _resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(target, "a", encoding="utf-8") as f:
            await f.write(content)
        return ToolResult.success(f"appended {len(content)} chars -> {target}")


class DeletePathTool(BaseTool):
    name = "delete_path"
    description = (
        "Delete a file or (with recursive=true) a directory. There is no trash — "
        "this is permanent. Returns the number of items removed."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "recursive": {"type": "boolean", "default": False},
        },
        "required": ["path"],
    }
    timeout_s = 30.0

    async def run(self, path: str, recursive: bool = False) -> ToolResult:
        target = _resolve(path)
        if not target.exists():
            return ToolResult.failure(f"path not found: {target}")
        import asyncio

        def _do() -> int:
            if target.is_dir():
                if not recursive:
                    raise RuntimeError("directory delete requires recursive=true")
                count = sum(1 for _ in target.rglob("*"))
                shutil.rmtree(target)
                return count + 1
            target.unlink()
            return 1

        removed = await asyncio.to_thread(_do)
        return ToolResult.success(f"deleted {target} ({removed} item(s))")


class StatPathTool(BaseTool):
    name = "stat_path"
    description = "Quick metadata probe: type, size, mtime, exists."
    parameters = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }
    timeout_s = 5.0

    async def run(self, path: str) -> ToolResult:
        target = _resolve(path)
        if not target.exists():
            return ToolResult.success(f"{target}\n  exists=false")
        st = target.stat()
        kind = "dir" if target.is_dir() else "file" if target.is_file() else "other"
        return ToolResult.success(
            f"{target}\n  exists=true\n  kind={kind}\n  size={st.st_size}\n  mtime={st.st_mtime:.0f}",
            meta={"kind": kind, "size": st.st_size, "mtime": st.st_mtime},
        )


async def _atomic_write_text(target: Path, text: str) -> None:
    import asyncio

    def _do() -> None:
        fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".krypton_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
                f.write(text)
            os.replace(tmp, target)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    await asyncio.to_thread(_do)


def tools() -> list[Any]:
    return [
        ReadFileTool(),
        WriteFileTool(),
        EditFileTool(),
        AppendFileTool(),
        DeletePathTool(),
        StatPathTool(),
    ]
