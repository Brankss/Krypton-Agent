"""Ensure the local Ollama daemon is running before the agent starts.

Behavior:
  1. Ping `{host}/api/version`. If 200 OK -> done.
  2. Otherwise locate the `ollama` binary in PATH and spawn `ollama serve`
     detached so it survives Krypton exiting.
  3. Poll the version endpoint until the daemon is ready (or timeout).

Returns True if Ollama is reachable when the function returns.
"""
from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
import time

import httpx


async def ensure_ollama(host: str, *, ready_timeout: float = 60.0, verbose: bool = True) -> bool:
    if await _ping(host):
        if verbose:
            print(f"[ollama] already running at {host}")
        return True

    exe = shutil.which("ollama")
    if exe is None:
        if verbose:
            print("[ollama] daemon not reachable AND `ollama` binary not found in PATH — install from https://ollama.com")
        return False

    if verbose:
        print(f"[ollama] daemon not running; launching `{exe} serve` detached...")

    _spawn_detached([exe, "serve"])

    # Poll until ready
    deadline = time.monotonic() + ready_timeout
    while time.monotonic() < deadline:
        if await _ping(host):
            if verbose:
                print(f"[ollama] ready at {host}")
            return True
        await asyncio.sleep(0.4)

    if verbose:
        print(f"[ollama] FAILED to come up within {ready_timeout:.0f}s")
    return False


async def _ping(host: str) -> bool:
    url = f"{host.rstrip('/')}/api/version"
    try:
        async with httpx.AsyncClient(timeout=1.5) as c:
            r = await c.get(url)
            return r.status_code == 200
    except (httpx.HTTPError, OSError):
        return False


def _spawn_detached(argv: list[str]) -> None:
    if sys.platform == "win32":
        # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_BREAKAWAY_FROM_JOB
        flags = 0x00000008 | 0x00000200 | 0x01000000
        subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=flags,
            close_fds=True,
        )
    else:
        subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
