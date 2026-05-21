"""Krypton launcher.

Run with:
    python main.py             # foreground (Ctrl+C to stop)
    python main.py --detach    # background — survives terminal close
    python main.py --stop      # kill any running instance and exit
    python main.py --cli       # local terminal session instead of Telegram
    python main.py --doctor    # health-check the provider
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = ROOT / ".venv" / "Scripts" / "python.exe"
if not PY.exists():
    PY = Path(sys.executable)


# ---------------------------------------------------------------------------


def _kill_existing() -> int:
    """Stop any python.exe running 'krypton telegram'. Returns number killed."""
    if sys.platform != "win32":
        return 0
    ps = (
        "$n = 0; "
        "Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" | "
        "Where-Object { $_.CommandLine -like '*krypton*telegram*' } | "
        "ForEach-Object { Stop-Process -Id $_.ProcessId -Force; $n++ }; "
        "Write-Output $n"
    )
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
            capture_output=True, text=True, timeout=10,
        )
        return int(r.stdout.strip() or "0")
    except Exception:  # noqa: BLE001
        return 0


def _run_foreground(subcommand: str) -> int:
    print(f"==> python -m krypton {subcommand}  (Ctrl+C to stop)")
    try:
        return subprocess.run([str(PY), "-m", "krypton", subcommand], cwd=str(ROOT)).returncode
    except KeyboardInterrupt:
        return 0


def _run_detached() -> None:
    log_dir = ROOT / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    out = open(log_dir / "telegram.log", "ab")
    err = open(log_dir / "telegram.err.log", "ab")

    flags = 0
    if sys.platform == "win32":
        # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_BREAKAWAY_FROM_JOB
        flags = 0x00000008 | 0x00000200 | 0x01000000

    proc = subprocess.Popen(
        [str(PY), "-m", "krypton", "telegram"],
        cwd=str(ROOT),
        stdin=subprocess.DEVNULL,
        stdout=out,
        stderr=err,
        creationflags=flags,
        close_fds=False,
        start_new_session=sys.platform != "win32",
    )
    print(f"Krypton bot detached  PID={proc.pid}")
    print(f"Logs : {log_dir / 'telegram.err.log'}")
    print("Stop : python main.py --stop")


# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(prog="main.py", description="Launch Krypton.")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--detach", action="store_true", help="Run bot in background")
    mode.add_argument("--stop",   action="store_true", help="Kill any running bot and exit")
    mode.add_argument("--cli",    action="store_true", help="Run interactive CLI instead of the bot")
    mode.add_argument("--doctor", action="store_true", help="Health-check the provider and exit")
    p.add_argument("--no-kill", action="store_true", help="Don't kill existing instances before launch")
    args = p.parse_args()

    if args.stop:
        n = _kill_existing()
        print(f"killed {n} running instance(s)")
        return 0

    if args.doctor:
        return _run_foreground("doctor")

    if args.cli:
        return _run_foreground("cli")

    # Default = telegram bot
    if not args.no_kill:
        n = _kill_existing()
        if n:
            print(f"killed {n} existing instance(s)")

    if args.detach:
        _run_detached()
        return 0
    return _run_foreground("telegram")


if __name__ == "__main__":
    sys.exit(main())
