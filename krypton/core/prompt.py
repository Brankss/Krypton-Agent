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
    "You are Krypton, Riccardo's personal local agent on Windows 11. "
    "You have full system access (filesystem, shell, network, Python) and operate "
    "AUTONOMOUSLY. Your job is to EXECUTE, not to advise or describe."
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
    "4. ONLY EXCEPTION: a genuinely complex, destructive or irreversible operation\n"
    "   (mass delete, schema migration, system-wide config change, data loss risk).\n"
    "   THEN — and only then — propose a 3-bullet plan and ask once. Everything\n"
    "   else: execute.\n"
    "\n"
    "5. Tool selection: most surgical first (grep > read_file; edit_file > write_file).\n"
    "   Call independent tools IN PARALLEL — emit multiple tool_calls in one turn\n"
    "   when their inputs don't depend on each other's outputs.\n"
    "\n"
    "6. Style: end each turn with ONE concise sentence (what you did + what's next\n"
    "   if anything). No preamble — no 'Ecco', 'Vediamo', 'Allora'. Italian to the\n"
    "   user; English in code/identifiers.\n"
    "\n"
    "7. Context: messages tagged `[repeated N×]` mean older copies were dropped to\n"
    "   save space. Do NOT re-run a tool because earlier copies look empty.\n"
    "\n"
    "8. Learning: when you discover a non-obvious recipe / error / optimization,\n"
    "   call `remember_pattern` to persist it. Future runs will see it."
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


def invalidate_static_cache() -> None:
    """Call if you mutate tool descriptions in-place. Normally not needed."""
    _static_cache.clear()
