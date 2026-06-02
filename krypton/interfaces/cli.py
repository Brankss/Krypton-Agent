"""Local terminal interface.

Rich-powered streaming output, slash commands for provider/model swap,
context reset, pattern listing.
"""
from __future__ import annotations

import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from krypton.bootstrap import build_agent
from krypton.config import ProviderName, settings
from krypton.core.agent import Agent
from krypton.tools.base import ToolResult


console = Console()


def _print_banner(agent: Agent) -> None:
    body = Text()
    body.append("Krypton Agent ", style="bold cyan")
    body.append(f"  provider={agent.provider.name}  model={agent.provider.model}\n", style="dim")
    body.append(f"workdir: {settings.workdir}\n", style="dim")
    body.append("Commands: /provider <name>  /model <name>  /status  /reset  /patterns  /quit\n", style="dim")
    console.print(Panel(body, border_style="cyan"))


async def _stream_text(buf: list[str], text: str) -> None:
    buf.append(text)
    sys.stdout.write(text)
    sys.stdout.flush()


async def _on_tool(name: str, args: dict, res: ToolResult) -> None:
    status = "[green]OK[/]" if res.ok else "[red]ERR[/]"
    arg_preview = ", ".join(f"{k}={_short(v)}" for k, v in args.items())
    console.print(f"\n[dim]· tool {status} {name}({arg_preview}) {res.duration_ms}ms[/]")


def _short(v) -> str:
    s = str(v)
    return s if len(s) < 60 else s[:57] + "..."


async def _read_line(prompt: str) -> str:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.patch_stdout import patch_stdout

    if not hasattr(_read_line, "_session"):
        _read_line._session = PromptSession()  # type: ignore[attr-defined]
    with patch_stdout():
        return await _read_line._session.prompt_async(prompt)  # type: ignore[attr-defined]


async def run_cli(initial_provider: ProviderName | None = None) -> None:
    agent = build_agent(initial_provider, interface="cli")
    _print_banner(agent)

    try:
        while True:
            try:
                user_text = (await _read_line("you ▸ ")).strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]bye[/]")
                return
            if not user_text:
                continue

            if user_text.startswith("/"):
                cont = await _handle_command(user_text, agent)
                if cont is None:
                    return
                if cont is True:
                    continue
                # cont is False => fall through and treat as text (unused)

            console.print("[bold magenta]krypton ▸[/] ", end="")
            buf: list[str] = []
            result = await agent.turn(
                user_text,
                on_text=lambda t, b=buf: _stream_text(b, t),
                on_tool=_on_tool,
            )
            sys.stdout.write("\n")
            if result.aborted and result.error:
                console.print(f"[red]error:[/] {result.error}")
            for art in result.artifacts:
                console.print(f"[dim]artifact:[/] {art.path}")
    finally:
        await agent.aclose()


async def _handle_command(line: str, agent: Agent) -> bool | None:
    """Returns True to continue REPL, None to quit, False if unhandled."""
    parts = line.split()
    cmd, rest = parts[0], parts[1:]

    if cmd in ("/quit", "/exit"):
        return None

    if cmd == "/reset":
        agent.context.reset()
        console.print("[dim]context cleared[/]")
        return True

    if cmd == "/provider":
        if not rest:
            console.print(f"current: {agent.provider.name}")
            return True
        try:
            from krypton.providers import build_provider
            new_p = build_provider(rest[0])  # type: ignore[arg-type]
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]could not switch: {e}[/]")
            return True
        await agent.provider.aclose()
        agent.provider = new_p
        console.print(f"[green]provider -> {new_p.name} ({new_p.model})[/]")
        return True

    if cmd == "/model":
        if not rest:
            console.print(f"current: {agent.provider.model}")
            return True
        agent.provider.model = rest[0]
        console.print(f"[green]model -> {agent.provider.model}[/]")
        return True

    if cmd == "/patterns":
        pats = await agent.memory.list_patterns(limit=50)
        if not pats:
            console.print("[dim]no patterns yet[/]")
        for p in pats:
            console.print(f"  #{p.id} [{p.kind}] {p.title}  [dim]uses={p.uses}[/]")
        return True

    if cmd == "/status":
        ctx = agent.context
        console.print(
            f"provider: {agent.provider.name}\n"
            f"model:    {agent.provider.model}\n"
            f"context:  ~{ctx.token_count} tokens, {ctx.entry_count} entries"
        )
        called = [(n, s) for n, s in agent.registry.stats().items() if s.calls]
        if called:
            console.print("tool calls:")
            for name, s in sorted(called, key=lambda kv: kv[1].calls, reverse=True):
                console.print(f"  {name}: {s.calls} call(s), {s.errors} err, avg {s.avg_ms:.0f}ms")
        else:
            console.print("[dim]no tool calls yet[/]")
        return True

    if cmd == "/help":
        console.print(
            Markdown(
                "**Commands**\n\n"
                "- `/provider <ollama_local|ollama_cloud|openrouter|nvidia>`\n"
                "- `/model <name>`\n"
                "- `/status` provider, model, context size + tool stats\n"
                "- `/reset` clear conversation\n"
                "- `/patterns` list learned patterns\n"
                "- `/quit` exit"
            )
        )
        return True

    console.print(f"[yellow]unknown command:[/] {cmd}")
    return True
