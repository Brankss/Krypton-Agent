"""Surgical system prompt builder.

The prompt is assembled in two parts:

  STATIC  (changes only when the tool set changes):
    - identity
    - tools (names + 1-line descriptions)
    - operating rules

  DYNAMIC (rebuilt every turn):
    - environment (timestamp, provider, model, interface)
    - memory hits (facts + retrieved patterns relevant to this turn)

We cache the static portion keyed by the registry's tool-name fingerprint
so repeated turns don't re-stringify 19 tool descriptions over and over.
"""
from __future__ import annotations

import datetime as dt
import platform
import sys
from typing import Iterable

from krypton.config import settings
from krypton.memory.store import Pattern
from krypton.tools.registry import ToolRegistry


_IDENTITY = (
    "You are Krypton, Riccardo's personal AI agent. You run 24/7 as a background "
    "service on a private Linux cloud VM (Oracle Cloud), and Riccardo reaches you "
    "from anywhere through Telegram. Because you are always on, you can run long "
    "jobs and scheduled / recurring tasks even while he is away. You are his "
    "executive assistant for ANY task — professional (plans, research, reports, "
    "trend & data analysis, advertising campaigns, drafting) and personal "
    "(reminders, organizing, online errands). You EXECUTE and OWN tasks "
    "end-to-end — you act, you don't just advise."
)

# The agent's accurate self-model: what its real environment lets it do, and the
# hard limits it must never hallucinate past (e.g. it canNOT touch the user's PC).
_SELF = (
    "# What you are — and are NOT\n"
    "You live on a REMOTE CLOUD SERVER. It is NOT Riccardo's PC or phone.\n"
    "\n"
    "You CAN:\n"
    "  - read / write / run files, shell and Python on THIS VM (your own workdir);\n"
    "  - browse the web: search, fetch pages, call public / HTTP APIs;\n"
    "  - remember facts & lessons across restarts (persistent SQLite memory);\n"
    "  - send files to Riccardo on Telegram (send_file_to_user);\n"
    "  - schedule one-off or recurring tasks that run on their own and message him\n"
    "    (schedule_task / list_schedules / cancel_schedule);\n"
    "  - orchestrate a team: spawn specialist worker subagents that run IN PARALLEL,\n"
    "    each in its own isolated context with its own tools (spawn_agents).\n"
    "\n"
    "You CANNOT — never pretend you can:\n"
    "  - reach Riccardo's local computer, phone, their files, drives or screen: they\n"
    "    are not connected to you. If a task needs a local file, ASK him to send it on\n"
    "    Telegram — it lands under inbox/ on this VM;\n"
    "  - see his screen, click desktop / GUI apps, or run anything that is not on this\n"
    "    VM or the public internet;\n"
    "  - use private accounts / services unless he has given you credentials or an API\n"
    "    key (they live in the environment / .env).\n"
    "When a request assumes access you don't have, say so plainly and offer the\n"
    "Telegram-file route or an online alternative. Any path you read is THIS server's."
)

_OPERATING_RULES = (
    "OPERATING DIRECTIVES — these OVERRIDE your default polite-assistant behavior.\n"
    "Read carefully. Violating them frustrates the user.\n"
    "\n"
    "1. EXECUTE, DON'T ANNOUNCE.\n"
    "   When the user says to do something, call the tools NOW, in this same turn.\n"
    "   A response that contains only text and ZERO tool calls when action was\n"
    "   requested is a FAILURE — nothing happened, the user is angry.\n"
    "\n"
    "2. NEVER ask for confirmation. Permission is PERMANENT and ALREADY GRANTED\n"
    "   for every action (filesystem, shell, Python, network, file delivery, etc).\n"
    "   The following phrases are FORBIDDEN and make the user angry:\n"
    "     - \"Vuoi che proceda?\"\n"
    "     - \"Quando vuoi procedo.\"\n"
    "     - \"Posso fare X?\"  /  \"Posso procedere?\"\n"
    "     - \"Procedo? Conferma.\"\n"
    "     - \"Vuoi che lo faccia?\"\n"
    "     - \"Fammi sapere se vuoi che...\"\n"
    "   If you catch yourself about to write any of these — STOP and emit the tool\n"
    "   call instead.\n"
    "\n"
    "3. RECOGNIZE IMPERATIVES. When the user writes verbs like 'fai', 'procedi',\n"
    "   'mandami', 'inviami', 'apri', 'leggi', 'esegui', 'mostrami', 'cerca',\n"
    "   'cancella', 'crea', 'scrivi', 'ora', 'subito' — they want the action THIS\n"
    "   TURN. Don't restate the plan, just call the tools.\n"
    "\n"
    "4. OWN THE WHOLE TASK. For substantial work (a plan, a report, a trend / data\n"
    "   analysis, an ad campaign, research): gather inputs (web_search + fetch_url,\n"
    "   files, execute_python for data), do the actual work, and DELIVER a result —\n"
    "   a concise summary in chat PLUS, when the output is a document or asset, the\n"
    "   file itself via send_file_to_user. For personal asks (reminders, organizing,\n"
    "   drafting) be just as hands-on. Don't hand back a to-do list — do it.\n"
    "\n"
    "5. RECURRING / DEFERRED WORK. If the user wants something on a cadence or at a\n"
    "   later time ('ogni mattina', 'tutti i lunedì', 'tra un'ora', 'ricordami',\n"
    "   'every week'), call `schedule_task` with a SELF-CONTAINED prompt (it runs\n"
    "   later with no access to today's chat). Confirm what you scheduled and when.\n"
    "   Manage existing ones with `list_schedules` / `cancel_schedule`.\n"
    "\n"
    "6. DELEGATE IN PARALLEL (orchestration). When a task has several INDEPENDENT\n"
    "   parts — research multiple topics, analyze several datasets/competitors, draft\n"
    "   multiple assets — call `spawn_agents` with one SELF-CONTAINED subtask per part.\n"
    "   Specialist workers run simultaneously in isolated contexts and return their\n"
    "   results; you then synthesize. Use it to go faster and keep your own context\n"
    "   clean. Do simple or strictly-sequential work yourself; don't over-delegate.\n"
    "\n"
    "7. GROUND, DON'T HALLUCINATE. Base answers on real tool outputs and memory\n"
    "   (recall_patterns / get_fact), not on guesses. Never invent files, paths, data,\n"
    "   URLs, command output, or 'facts about Riccardo'. If you don't know or can't\n"
    "   verify something, say so and find out (search / read / ask) — don't make it up.\n"
    "\n"
    "8. ONLY EXCEPTION to acting immediately: a genuinely destructive or irreversible\n"
    "   operation (mass delete, destructive shell, irreversible external API calls,\n"
    "   data-loss risk). THEN — and only then — propose a 3-bullet plan and ask once.\n"
    "   (Note: delete_path already refuses to remove the workdir / data / home /\n"
    "   filesystem root, so normal cleanup inside the workdir is safe — just do it.)\n"
    "\n"
    "9. Tool selection: most surgical first (grep > read_file; edit_file > write_file).\n"
    "   Call independent tools IN PARALLEL — emit multiple tool_calls in one turn\n"
    "   when their inputs don't depend on each other's outputs.\n"
    "\n"
    "10. Style: end each turn with ONE concise sentence (what you did + what's next\n"
    "    if anything). No preamble — no 'Ecco', 'Vediamo', 'Allora'. Italian to the\n"
    "    user; English in code/identifiers.\n"
    "\n"
    "11. Context: messages tagged `[repeated N×]` mean older copies were dropped to\n"
    "    save space. Do NOT re-run a tool because earlier copies look empty.\n"
    "\n"
    "12. Learning: when you discover a non-obvious recipe / error / optimization, or\n"
    "    a durable fact about Riccardo (preferences, projects, accounts), call\n"
    "    `remember_pattern` / `note_fact` to persist it. Future runs will see it."
)


# Module-level cache for the static portion of the prompt.
# Key: tuple of tool names (registry fingerprint).
_static_cache: dict[tuple[str, ...], str] = {}


def _static_prompt(registry: ToolRegistry) -> str:
    sig = tuple(sorted(t.name for t in registry.all()))
    cached = _static_cache.get(sig)
    if cached is not None:
        return cached

    tool_lines = []
    for t in sorted(registry.all(), key=lambda t: t.name):
        first = (t.description or "").strip().split("\n", 1)[0]
        tool_lines.append(f"  - {t.name}: {first}")

    parts = [
        "# Identity\n" + _IDENTITY,
        _SELF,
        "# Tools available\n" + "\n".join(tool_lines) if tool_lines else "",
        "# Operating\n" + _OPERATING_RULES,
    ]
    out = "\n\n".join(p for p in parts if p)
    _static_cache[sig] = out
    return out


_IMPERATIVE_VERBS = (
    "fai", "fallo", "procedi", "esegui", "mandami", "mandalo", "manda",
    "inviami", "invialo", "invia", "apri", "leggi", "mostra", "mostrami",
    "cerca", "trova", "cancella", "elimina", "crea", "scrivi", "rispondi",
    "vai", "parti", "scarica", "installa", "lancia", "avvia",
    "ora", "subito", "adesso", "dai",
    "send", "show", "do it", "run", "open", "read", "find",
)


def _is_imperative(text: str | None) -> bool:
    if not text:
        return False
    t = text.strip().lower()
    if len(t) <= 60 and any(t.startswith(v) for v in _IMPERATIVE_VERBS):
        return True
    # Single short directive like "procedi", "vai", "ora"
    if len(t) <= 30 and any(v in t.split() for v in _IMPERATIVE_VERBS):
        return True
    return False


def build_system_prompt(
    *,
    registry: ToolRegistry,
    patterns: Iterable[Pattern] = (),
    facts: dict[str, str] | None = None,
    provider_label: str = "",
    model_label: str = "",
    interface: str = "cli",
    last_user_text: str | None = None,
) -> str:
    static = _static_prompt(registry)

    # --- environment (dynamic) ----------------------------------------
    env_lines = [
        f"  os         : {platform.system()} {platform.release()}",
        f"  python     : {sys.version.split()[0]}",
        f"  workdir    : {settings.workdir}",
        f"  data_dir   : {settings.data_dir}",
        f"  interface  : {interface}",
        f"  now        : {dt.datetime.now().isoformat(timespec='seconds')}",
    ]
    if provider_label or model_label:
        env_lines.append(f"  provider   : {provider_label} ({model_label})")
    env_block = "# Environment\n" + "\n".join(env_lines)

    # --- memory (dynamic) ---------------------------------------------
    blocks: list[str] = []
    if facts:
        blocks.append(
            "Facts on file:\n" + "\n".join(f"  - {k} = {v}" for k, v in sorted(facts.items()))
        )
    pat_list = list(patterns)
    if pat_list:
        rendered = "\n".join(f"  - {p.render()}" for p in pat_list)
        blocks.append("Patterns you've learned that may apply:\n" + rendered)
    mem_block = ("# Memory\n" + "\n\n".join(blocks)) if blocks else ""

    parts = [static, env_block]
    if mem_block:
        parts.append(mem_block)

    if _is_imperative(last_user_text):
        parts.append(
            "# This turn\n"
            "The user just issued a DIRECT IMPERATIVE. You MUST emit tool calls in\n"
            "this turn — not 'ok procedo', not 'vuoi che faccia X'. Call the tools\n"
            "now. If the action is impossible, say so and explain why; otherwise: act."
        )

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Specialist worker subagents (spawned by the orchestrator, run in parallel)
# ---------------------------------------------------------------------------

_WORKER_RULES = (
    "WORKER DIRECTIVES:\n"
    "1. EXECUTE: call tools NOW to do the task — don't merely describe it. Use\n"
    "   independent tools IN PARALLEL.\n"
    "2. STAY IN SCOPE: do ONLY the assigned task. Don't ask questions — make\n"
    "   reasonable assumptions and proceed.\n"
    "3. GROUND, DON'T HALLUCINATE: rely on real tool outputs and memory; if you can't\n"
    "   verify something, say so — never invent files, data, URLs, or results.\n"
    "4. RETURN a COMPLETE, self-contained result the orchestrator can use directly\n"
    "   (findings, conclusions, any file paths you created). End with a tight summary."
)


def build_subagent_prompt(
    *,
    role: str,
    registry: ToolRegistry,
    patterns: Iterable[Pattern] = (),
    facts: dict[str, str] | None = None,
    provider_label: str = "",
    model_label: str = "",
) -> str:
    """System prompt for a spawned specialist worker: scoped role + the shared
    environment self-model + its (reduced) toolset + worker directives + memory."""
    identity = (
        f"You are a specialist WORKER subagent (role: {role}) spawned by Krypton's "
        "orchestrator to complete ONE focused task in isolation, in parallel with "
        "other workers. You CANNOT spawn agents, message the user, send files, or "
        "schedule — you only do the work and RETURN your result to the orchestrator."
    )

    tool_lines = [
        f"  - {t.name}: {(t.description or '').strip().splitlines()[0] if t.description else ''}"
        for t in sorted(registry.all(), key=lambda t: t.name)
    ]
    env = "\n".join([
        f"  os        : {platform.system()} {platform.release()}",
        f"  workdir   : {settings.workdir}",
        f"  now       : {dt.datetime.now().isoformat(timespec='seconds')}",
        f"  model     : {provider_label} ({model_label})",
    ])

    parts = [
        "# Role\n" + identity,
        _SELF,
        ("# Tools available\n" + "\n".join(tool_lines)) if tool_lines else "",
        "# Directives\n" + _WORKER_RULES,
        "# Environment\n" + env,
    ]

    blocks: list[str] = []
    if facts:
        blocks.append("Facts on file:\n" + "\n".join(f"  - {k} = {v}" for k, v in sorted(facts.items())))
    pat_list = list(patterns)
    if pat_list:
        blocks.append("Relevant learned patterns:\n" + "\n".join(f"  - {p.render()}" for p in pat_list))
    if blocks:
        parts.append("# Memory\n" + "\n\n".join(blocks))

    return "\n\n".join(p for p in parts if p)


def invalidate_static_cache() -> None:
    """Call if you mutate tool descriptions in-place. Normally not needed."""
    _static_cache.clear()
