"""Entry point.

Usage:
    python -m krypton            # launches interactive CLI
    python -m krypton cli         # same as above
    python -m krypton telegram    # runs the Telegram bot
    python -m krypton doctor      # health-checks providers + tools
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from krypton.config import settings
from krypton.runtime import ensure_ollama


async def _maybe_boot_ollama(provider_override: str | None = None) -> None:
    """If the chosen provider needs the local Ollama daemon, make sure it's up."""
    chosen = provider_override or settings.provider
    if chosen == "ollama_local":
        await ensure_ollama(settings.ollama_local_host)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="krypton", description="Krypton local agent.")
    sub = p.add_subparsers(dest="cmd")

    cli = sub.add_parser("cli", help="interactive terminal session")
    cli.add_argument(
        "--provider",
        choices=["ollama_local", "ollama_cloud", "openrouter", "nvidia"],
        help="override KRYPTON_PROVIDER for this session",
    )

    sub.add_parser("telegram", help="run the Telegram bot")
    sub.add_parser("doctor", help="run health checks (provider reachability, etc.)")
    return p


async def _run_cli(args: argparse.Namespace) -> int:
    await _maybe_boot_ollama(args.provider)
    from krypton.interfaces.cli import run_cli
    await run_cli(args.provider)
    return 0


def _run_telegram(_args: argparse.Namespace) -> int:
    asyncio.run(_maybe_boot_ollama())
    from krypton.interfaces.telegram import run_telegram
    run_telegram()
    return 0


async def _run_doctor(_args: argparse.Namespace) -> int:
    await _maybe_boot_ollama()
    from krypton.providers import build_provider
    from krypton.providers.base import Message

    print(f"workdir       : {settings.workdir}")
    print(f"data_dir      : {settings.data_dir}")
    print(f"active provider: {settings.provider}")
    print("--- pinging provider ---")
    try:
        p = build_provider()
        seen = ""
        async for ev in p.stream([Message(role="user", content="ping")], temperature=0.0, max_tokens=8):
            from krypton.providers.base import TextDelta, Done
            if isinstance(ev, TextDelta):
                seen += ev.text
            elif isinstance(ev, Done):
                break
        await p.aclose()
        print(f"  reply: {seen.strip()!r}")
        print("  OK")
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL: {type(e).__name__}: {e}")
        return 1
    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    cmd = args.cmd or "cli"

    if cmd == "cli":
        return asyncio.run(_run_cli(args))
    if cmd == "telegram":
        return _run_telegram(args)
    if cmd == "doctor":
        return asyncio.run(_run_doctor(args))
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
