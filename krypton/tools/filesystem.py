"""Fast filesystem tools: list / glob / grep.

All heavy walking happens inside an asyncio executor so we don't pin
the event loop. We use `os.scandir` (the fastest stat-batched API on
Windows + POSIX), prune junk directories aggressively, and cap result
sizes so the LLM never drowns in output.
"""
from __future__ import annotations

import asyncio
import fnmatch
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from krypton.config import settings
from krypton.tools.base import BaseTool, ToolResult

# Directory names we never descend into unless the user opts in.
_PRUNE_DIRS = frozenset(
    {
        ".git", ".hg", ".svn",
        "node_modules", "bower_components",
        "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
        ".venv", "venv", "env", ".tox",
        "dist", "build", ".next", ".nuxt", ".cache",
        ".idea", ".vscode",
        "target",  # rust / java
        "Pods",    # ios
    }
)

# Files we silently skip when grepping by default (binaries / huge).
_BINARY_EXT = frozenset(
    {
        ".exe", ".dll", ".so", ".dylib", ".bin", ".o", ".a", ".lib",
        ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".ico", ".tif", ".tiff",
        ".mp3", ".mp4", ".wav", ".mov", ".mkv", ".avi", ".flac", ".ogg",
        ".zip", ".tar", ".gz", ".7z", ".rar", ".xz", ".bz2",
        ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
        ".pyc", ".pyd", ".class", ".jar",
        ".db", ".sqlite", ".sqlite3",
    }
)


def _resolve(path: str | None) -> Path:
    if not path:
        return settings.workdir
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = settings.workdir / p
    return p


def _should_prune(name: str, allow_hidden: bool) -> bool:
    if name in _PRUNE_DIRS:
        return True
    if not allow_hidden and name.startswith("."):
        return True
    return False


# ===========================================================================
# list_directory
# ===========================================================================


class ListDirectoryTool(BaseTool):
    name = "list_directory"
    description = (
        "Lists immediate contents of a directory with type, size and mtime. "
        "Returns a compact table — use this before reading or globbing."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute or relative path. Defaults to workdir."},
            "show_hidden": {"type": "boolean", "default": False},
            "limit": {"type": "integer", "default": 200, "description": "Max entries to return."},
        },
    }
    timeout_s = 10.0

    async def run(self, path: str | None = None, show_hidden: bool = False, limit: int = 200) -> ToolResult:
        target = _resolve(path)
        if not target.exists():
            return ToolResult.failure(f"path does not exist: {target}")
        if not target.is_dir():
            return ToolResult.failure(f"not a directory: {target}")

        def _scan() -> tuple[list[str], int]:
            rows: list[tuple[str, str, int, float]] = []
            total = 0
            with os.scandir(target) as it:
                for entry in it:
                    total += 1
                    if not show_hidden and entry.name.startswith("."):
                        continue
                    try:
                        st = entry.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    kind = "DIR" if entry.is_dir(follow_symlinks=False) else "FILE"
                    rows.append((entry.name, kind, st.st_size, st.st_mtime))
            rows.sort(key=lambda r: (r[1] != "DIR", r[0].lower()))
            shown = rows[:limit]
            lines = [f"{target}  ({len(rows)} entries, showing {len(shown)})"]
            for name, kind, size, _ in shown:
                size_s = "-" if kind == "DIR" else _human_size(size)
                lines.append(f"  {kind:<4} {size_s:>9}  {name}")
            return lines, total

        lines, total = await asyncio.to_thread(_scan)
        return ToolResult.success("\n".join(lines), meta={"total_entries": total})


# ===========================================================================
# find_files (glob walker)
# ===========================================================================


class FindFilesTool(BaseTool):
    name = "find_files"
    description = (
        "Recursive glob walker. Use for 'find every *.py under src/' style queries. "
        "Prunes junk dirs (node_modules, .git, __pycache__, ...) automatically. "
        "Returns paths sorted by mtime (newest first)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob like '*.py' or 'test_*.ts'. Required."},
            "path": {"type": "string", "description": "Root to walk. Defaults to workdir."},
            "max_depth": {"type": "integer", "default": 12},
            "limit": {"type": "integer", "default": 300},
            "include_hidden": {"type": "boolean", "default": False},
        },
        "required": ["pattern"],
    }
    timeout_s = 30.0

    async def run(
        self,
        pattern: str,
        path: str | None = None,
        max_depth: int = 12,
        limit: int = 300,
        include_hidden: bool = False,
    ) -> ToolResult:
        root = _resolve(path)
        if not root.is_dir():
            return ToolResult.failure(f"not a directory: {root}")

        def _walk() -> list[tuple[str, float]]:
            found: list[tuple[str, float]] = []
            root_str = str(root)
            root_len = len(root_str.rstrip(os.sep)) + 1
            stack: list[tuple[str, int]] = [(root_str, 0)]
            while stack and len(found) < limit * 4:
                current, depth = stack.pop()
                try:
                    it = os.scandir(current)
                except (PermissionError, FileNotFoundError, OSError):
                    continue
                with it:
                    for entry in it:
                        try:
                            is_dir = entry.is_dir(follow_symlinks=False)
                        except OSError:
                            continue
                        if is_dir:
                            if depth >= max_depth:
                                continue
                            if _should_prune(entry.name, include_hidden):
                                continue
                            stack.append((entry.path, depth + 1))
                            continue
                        if not include_hidden and entry.name.startswith("."):
                            continue
                        if fnmatch.fnmatch(entry.name, pattern):
                            try:
                                st = entry.stat(follow_symlinks=False)
                            except OSError:
                                continue
                            found.append((entry.path, st.st_mtime))
            found.sort(key=lambda r: r[1], reverse=True)
            return found[:limit]

        results = await asyncio.to_thread(_walk)
        if not results:
            return ToolResult.success(f"no matches for {pattern!r} under {root}")
        header = f"{len(results)} match(es) for {pattern!r} under {root} (newest first):"
        body = "\n".join(p for p, _ in results)
        return ToolResult.success(f"{header}\n{body}", meta={"count": len(results)})


# ===========================================================================
# grep
# ===========================================================================


@dataclass(slots=True)
class _Hit:
    path: str
    line_no: int
    line: str


class GrepTool(BaseTool):
    name = "grep"
    description = (
        "Regex search across files (ripgrep-like). Returns matching lines with "
        "file:line:text. Streams scandir + reads files concurrently with a bounded "
        "pool — never loads big binaries."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Python regex."},
            "path": {"type": "string", "description": "Root dir or single file. Defaults to workdir."},
            "glob": {"type": "string", "description": "Optional filename glob, e.g. '*.py'."},
            "case_insensitive": {"type": "boolean", "default": False},
            "max_results": {"type": "integer", "default": 200},
            "max_depth": {"type": "integer", "default": 12},
            "include_hidden": {"type": "boolean", "default": False},
            "context": {"type": "integer", "default": 0, "description": "Lines of context before/after."},
        },
        "required": ["pattern"],
    }
    timeout_s = 45.0

    async def run(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        case_insensitive: bool = False,
        max_results: int = 200,
        max_depth: int = 12,
        include_hidden: bool = False,
        context: int = 0,
    ) -> ToolResult:
        root = _resolve(path)
        if not root.exists():
            return ToolResult.failure(f"path does not exist: {root}")
        try:
            rx = re.compile(pattern, re.IGNORECASE if case_insensitive else 0)
        except re.error as e:
            return ToolResult.failure(f"invalid regex: {e}")

        targets: Iterable[Path]
        if root.is_file():
            targets = [root]
        else:
            targets = _iter_files(root, glob=glob, max_depth=max_depth, include_hidden=include_hidden)

        sem = asyncio.Semaphore(64)  # SSDs love concurrent small reads
        hits: list[_Hit] = []
        lock = asyncio.Lock()
        done = asyncio.Event()

        async def _scan_file(p: Path) -> None:
            if done.is_set():
                return
            async with sem:
                if done.is_set():
                    return
                local = await asyncio.to_thread(_grep_file, p, rx, max_results, context)
                if not local:
                    return
                async with lock:
                    remaining = max_results - len(hits)
                    if remaining > 0:
                        hits.extend(local[:remaining])
                    if len(hits) >= max_results:
                        done.set()

        await asyncio.gather(*[_scan_file(p) for p in targets])

        if not hits:
            return ToolResult.success(f"no matches for /{pattern}/ under {root}")
        lines = [f"{h.path}:{h.line_no}: {h.line}" for h in hits]
        truncated = " (truncated)" if len(hits) >= max_results else ""
        header = f"{len(hits)} match(es){truncated} for /{pattern}/ under {root}:"
        return ToolResult.success(header + "\n" + "\n".join(lines), meta={"hits": len(hits)})


def _iter_files(
    root: Path,
    *,
    glob: str | None,
    max_depth: int,
    include_hidden: bool,
) -> list[Path]:
    out: list[Path] = []
    stack: list[tuple[str, int]] = [(str(root), 0)]
    while stack:
        cur, depth = stack.pop()
        try:
            it = os.scandir(cur)
        except (PermissionError, FileNotFoundError, OSError):
            continue
        with it:
            for entry in it:
                try:
                    is_dir = entry.is_dir(follow_symlinks=False)
                except OSError:
                    continue
                if is_dir:
                    if depth >= max_depth or _should_prune(entry.name, include_hidden):
                        continue
                    stack.append((entry.path, depth + 1))
                    continue
                if not include_hidden and entry.name.startswith("."):
                    continue
                ext = os.path.splitext(entry.name)[1].lower()
                if ext in _BINARY_EXT:
                    continue
                if glob and not fnmatch.fnmatch(entry.name, glob):
                    continue
                out.append(Path(entry.path))
    return out


def _grep_file(path: Path, rx: "re.Pattern[str]", max_results: int, context: int) -> list[_Hit]:
    """Search a single file.

      - skip binaries via a NUL-byte sniff on the first 1 KB
      - cap at 4 MB per file (configurable in code)
      - context=0  -> stream line-by-line (no full-file load)
      - context>0  -> load lines once (need lookback/lookahead anyway)
    """
    try:
        st = path.stat()
    except OSError:
        return []
    if st.st_size > 4 * 1024 * 1024:
        return []
    try:
        with open(path, "rb") as fb:
            if b"\x00" in fb.read(1024):
                return []
    except OSError:
        return []

    spath = str(path)
    hits: list[_Hit] = []

    if context <= 0:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f, start=1):
                    if rx.search(line):
                        hits.append(_Hit(spath, i, line.rstrip("\n")))
                        if len(hits) >= max_results:
                            break
        except OSError:
            pass
        return hits

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except OSError:
        return hits
    for i, line in enumerate(lines, start=1):
        if rx.search(line):
            lo = max(0, i - 1 - context)
            hi = min(len(lines), i + context)
            snippet = "".join(lines[lo:hi]).rstrip("\n")
            hits.append(_Hit(spath, i, snippet))
            if len(hits) >= max_results:
                break
    return hits


def _human_size(n: int) -> str:
    for unit in ("B", "K", "M", "G", "T"):
        if n < 1024:
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}P"


# ---------------------------------------------------------------------------


def tools() -> list[Any]:
    return [ListDirectoryTool(), FindFilesTool(), GrepTool()]
