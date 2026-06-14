"""High-power execution tools (Linux cloud VM).

  execute_python — runs arbitrary Python in a subprocess of the current
                   interpreter; stdout/stderr captured, full access to THIS VM.

  shell          — runs a shell command via /bin/sh -c (cmd.exe on Windows,
                   kept for dev parity). The agent's everyday Unix toolbox.

Per requirement: the agent is not sandboxed on its own VM. We do *not* gate
these behind interactive confirmation — the user explicitly asked for full
autonomy. We still cap output size so a runaway never destroys context.
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from typing import Any

from krypton.config import settings
from krypton.tools.base import BaseTool, ToolResult

_MAX_OUTPUT_BYTES = 64 * 1024  # 64 KB per stream


async def _run_subprocess(
    argv: list[str],
    *,
    cwd: str | None = None,
    timeout: float = 120.0,
    stdin: str | None = None,
    env: dict[str, str] | None = None,
) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.PIPE if stdin is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd or str(settings.workdir),
        env={**os.environ, **(env or {})},
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=stdin.encode("utf-8") if stdin else None),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        proc.kill()
        try:
            await proc.wait()
        except Exception:  # noqa: BLE001
            pass
        return 124, "", f"timeout after {timeout:.1f}s"
    return proc.returncode or 0, _decode_capped(stdout), _decode_capped(stderr)


def _decode_capped(b: bytes) -> str:
    if len(b) > _MAX_OUTPUT_BYTES:
        head = b[: _MAX_OUTPUT_BYTES // 2].decode("utf-8", errors="replace")
        tail = b[-_MAX_OUTPUT_BYTES // 2 :].decode("utf-8", errors="replace")
        return f"{head}\n... [truncated {len(b) - _MAX_OUTPUT_BYTES} bytes] ...\n{tail}"
    return b.decode("utf-8", errors="replace")


def _format(rc: int, out: str, err: str, *, label: str = "") -> str:
    parts: list[str] = []
    if label:
        parts.append(label)
    parts.append(f"exit_code: {rc}")
    if out.strip():
        parts.append(f"--- stdout ---\n{out.rstrip()}")
    if err.strip():
        parts.append(f"--- stderr ---\n{err.rstrip()}")
    if not out.strip() and not err.strip():
        parts.append("(no output)")
    return "\n".join(parts)


# ===========================================================================
# execute_python
# ===========================================================================


class ExecutePythonTool(BaseTool):
    name = "execute_python"
    description = (
        "Execute arbitrary Python code in a fresh subprocess of the current interpreter. "
        "Has full filesystem and network access. Use this whenever you'd reach for a one-off "
        "script: data munging, parsing files, calling APIs, plotting, etc. Stdout/stderr captured."
    )
    parameters = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "The Python source to run."},
            "timeout": {"type": "number", "default": 120.0},
            "cwd": {"type": "string", "description": "Working directory. Defaults to KRYPTON_WORKDIR."},
        },
        "required": ["code"],
    }
    timeout_s = 600.0

    async def run(self, code: str, timeout: float = 120.0, cwd: str | None = None) -> ToolResult:
        # write to a temp file so tracebacks have a real file name + line numbers
        fd, tmp = tempfile.mkstemp(prefix="krypton_exec_", suffix=".py")
        try:
            os.write(fd, code.encode("utf-8"))
        finally:
            os.close(fd)
        try:
            # -I: isolated mode (skip user-site/PYTHON* env discovery — faster startup)
            # -X utf8: force UTF-8 everywhere (Windows defaults to cp1252)
            # -u: unbuffered stdout/stderr (we want output immediately)
            rc, out, err = await _run_subprocess(
                [sys.executable, "-I", "-X", "utf8", "-u", tmp],
                cwd=cwd,
                timeout=timeout,
            )
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        ok = rc == 0
        return ToolResult(ok=ok, content=_format(rc, out, err, label="python"))


# ===========================================================================
# shell  —  the agent's Unix command line on its own VM
# ===========================================================================


class ShellTool(BaseTool):
    name = "shell"
    description = (
        "Run a shell command on this Linux VM via /bin/sh -c (cmd.exe on Windows). "
        "Your everyday Unix toolbox: ls, cat, curl, grep, package installs (apt/dnf), "
        "git, etc. Use for one-liners; reach for execute_python for real logic."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "timeout": {"type": "number", "default": 120.0},
            "cwd": {"type": "string"},
        },
        "required": ["command"],
    }
    timeout_s = 600.0

    async def run(self, command: str, timeout: float = 120.0, cwd: str | None = None) -> ToolResult:
        if sys.platform == "win32":
            argv = ["cmd.exe", "/d", "/s", "/c", command]
        else:
            argv = ["/bin/sh", "-c", command]
        rc, out, err = await _run_subprocess(argv, cwd=cwd, timeout=timeout)
        return ToolResult(ok=rc == 0, content=_format(rc, out, err, label="shell"))


def tools() -> list[Any]:
    return [ExecutePythonTool(), ShellTool()]
