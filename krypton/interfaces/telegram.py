"""Telegram bot interface.

Talks to Krypton from any chat. Restricts itself to user IDs listed in
TELEGRAM_AUTHORIZED_USER_IDS (other senders are silently ignored — a
warning is logged at startup if the allowlist is empty so you notice).

Handles inbound:
  * text       -> normal agent turn
  * documents  -> downloaded to workdir/inbox/, agent told the path
  * photos     -> downloaded to workdir/inbox/, agent told the path
  * voice      -> downloaded to workdir/inbox/ as .ogg, agent told the path
                  (Telegram Premium transcription is forwarded when present)

Special tool: `send_file_to_user` — only registered inside this
interface — lets the agent push a file back to the user's chat.

Slash commands:
  /start /status /reset /reboot /stop
  /provider <name>  /model <name>
"""
from __future__ import annotations

import asyncio
import datetime as dt
import logging
import re
from pathlib import Path
from typing import Any

from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from krypton.bootstrap import build_agent
from krypton.config import settings
from krypton.core.agent import Agent
from krypton.tools.base import BaseTool, ToolArtifact, ToolResult

log = logging.getLogger("krypton.telegram")


# ---------------------------------------------------------------------------
# Per-chat session — every chat keeps its own agent + conversation context.
# ---------------------------------------------------------------------------


class _Session:
    def __init__(self, agent: Agent, outbox: "asyncio.Queue[Path]") -> None:
        self.agent = agent
        self.outbox = outbox
        self.lock = asyncio.Lock()
        self.restored = False
        # Cancellable handle for the in-flight turn; /stop sets the event.
        self.current_task: asyncio.Task | None = None
        self.stop_event: asyncio.Event = asyncio.Event()


_SESSIONS: dict[int, _Session] = {}


def _get_session(chat_id: int) -> _Session:
    sess = _SESSIONS.get(chat_id)
    if sess is None:
        outbox: asyncio.Queue[Path] = asyncio.Queue()
        send_tool = SendFileToUserTool(outbox)
        agent = build_agent(interface="telegram", extra_tools=[send_tool])
        sess = _Session(agent=agent, outbox=outbox)
        _SESSIONS[chat_id] = sess
    return sess


async def _ensure_restored(sess: _Session, chat_id: int) -> None:
    if sess.restored:
        return
    sess.restored = True
    payload = await sess.agent.memory.load_conversation(chat_id)
    if not payload:
        return
    try:
        n = sess.agent.context.restore(payload)
        log.info("restored %d messages for chat %d", n, chat_id)
    except Exception as e:  # noqa: BLE001
        log.warning("failed to restore conversation for chat %d: %s", chat_id, e)


# ---------------------------------------------------------------------------
# Tool: send_file_to_user
# ---------------------------------------------------------------------------


class SendFileToUserTool(BaseTool):
    name = "send_file_to_user"
    description = (
        "Send a file from disk to the user's Telegram chat. CALL THIS IMMEDIATELY "
        "when the user asks for a file ('mandami', 'inviami', 'send me', 'manda il', etc.) — "
        "do NOT ask for confirmation, do NOT ask which one if the context makes it obvious. "
        "If you need to discover candidates first, call find_files/list_directory in the SAME "
        "turn and then call this. Max 50 MB per file (Telegram limit)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute or workdir-relative path."},
            "caption": {"type": "string", "default": ""},
        },
        "required": ["path"],
    }
    timeout_s = 5.0

    def __init__(self, outbox: "asyncio.Queue[Path]") -> None:
        self._outbox = outbox

    async def run(self, path: str, caption: str = "") -> ToolResult:
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = settings.workdir / p
        if not p.is_file():
            return ToolResult.failure(f"not a file: {p}")
        if p.stat().st_size > 50 * 1024 * 1024:
            return ToolResult.failure(f"file too large for Telegram (>50MB): {p}")
        await self._outbox.put(p)
        return ToolResult(
            ok=True,
            content=f"queued {p.name} ({p.stat().st_size} bytes) for delivery",
            artifacts=[ToolArtifact(path=str(p), label=caption or p.name)],
        )


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def _authorized(update: Update) -> bool:
    allowed = settings.authorized_telegram_ids
    if not allowed:
        return True  # no allowlist configured = open (warned at startup)
    uid = update.effective_user.id if update.effective_user else None
    return uid in allowed


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------


_HELP = (
    "Krypton online.\n\n"
    "Send me text, files, photos, or voice notes.\n\n"
    "Commands:\n"
    "  /status   show provider / model / context size\n"
    "  /reset    clear conversation memory (RAM + DB)\n"
    "  /stop     interrupt the current turn\n"
    "  /reboot   restart the bot process\n"
    "  /provider <ollama_local|ollama_cloud|openrouter>\n"
    "  /model <name>"
)


async def _cmd_start(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorized(update):
        return
    await update.message.reply_text(_HELP)


async def _cmd_reset(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorized(update):
        return
    chat_id = update.effective_chat.id
    sess = _get_session(chat_id)
    sess.agent.context.reset()
    await sess.agent.memory.clear_conversation(chat_id)
    await update.message.reply_text("context cleared (both in-memory and persisted).")


async def _cmd_stop(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorized(update):
        return
    sess = _get_session(update.effective_chat.id)
    task = sess.current_task
    if task and not task.done():
        sess.stop_event.set()
        task.cancel()
        await update.message.reply_text("⏹ stop signal sent; cancelling current turn.")
    else:
        await update.message.reply_text("nothing to stop — no turn in flight.")


# Convenience aliases — what the user types → canonical provider name.
_PROVIDER_ALIASES = {
    "ollama":       "ollama_local",
    "local":        "ollama_local",
    "ollama_local": "ollama_local",
    "cloud":        "ollama_cloud",
    "ollama_cloud": "ollama_cloud",
    "router":       "openrouter",
    "or":           "openrouter",
    "openrouter":   "openrouter",
    "nim":          "nvidia",
    "nvidia":       "nvidia",
}

_PROVIDER_HELP = (
    "usage: /provider <name>\n"
    "available:\n"
    "  ollama_local  (aliases: ollama, local)\n"
    "  ollama_cloud  (alias: cloud)\n"
    "  openrouter    (aliases: router, or)\n"
    "  nvidia        (alias: nim)"
)


async def _cmd_provider(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorized(update):
        return
    sess = _get_session(update.effective_chat.id)
    if not ctx.args:
        await update.message.reply_text(
            f"current: {sess.agent.provider.name}\n\n{_PROVIDER_HELP}"
        )
        return

    raw = ctx.args[0].strip().lower()
    canonical = _PROVIDER_ALIASES.get(raw)
    if canonical is None:
        await update.message.reply_text(
            f"unknown provider: {raw!r}\n\n{_PROVIDER_HELP}"
        )
        return

    from krypton.providers import build_provider
    try:
        new_p = build_provider(canonical)  # type: ignore[arg-type]
    except Exception as e:  # noqa: BLE001
        await update.message.reply_text(f"could not switch: {e}")
        return
    await sess.agent.provider.aclose()
    sess.agent.provider = new_p
    await update.message.reply_text(f"provider -> {new_p.name} ({new_p.model})")


async def _cmd_model(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorized(update):
        return
    sess = _get_session(update.effective_chat.id)
    if not ctx.args:
        await update.message.reply_text(f"current: {sess.agent.provider.model}")
        return
    sess.agent.provider.model = ctx.args[0]
    await update.message.reply_text(f"model -> {sess.agent.provider.model}")


async def _cmd_status(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorized(update):
        return
    sess = _get_session(update.effective_chat.id)
    in_flight = bool(sess.current_task and not sess.current_task.done())
    await update.message.reply_text(
        f"provider:  {sess.agent.provider.name}\n"
        f"model:     {sess.agent.provider.model}\n"
        f"context:   ~{sess.agent.context.token_count} tokens, "
        f"{len(sess.agent.context._entries)} entries\n"  # type: ignore[attr-defined]
        f"in-flight: {in_flight}"
    )


async def _cmd_reasoning(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Set chain-of-thought reasoning effort (NVIDIA provider).

    Usage:
        /reasoning                      -> show current level
        /reasoning none|low|medium|high -> set level

    Aliases: 'off' = 'none', 'on' = 'low'
    """
    if not _authorized(update):
        return
    sess = _get_session(update.effective_chat.id)
    prov = sess.agent.provider
    if not hasattr(prov, "reasoning_effort"):
        await update.message.reply_text(
            f"provider '{prov.name}' has no reasoning_effort knob. "
            f"Switch with /provider nvidia first."
        )
        return
    if not ctx.args:
        cur = prov.reasoning_effort or "default"  # type: ignore[attr-defined]
        await update.message.reply_text(f"reasoning_effort: {cur}")
        return

    arg = ctx.args[0].strip().lower()
    # convenience aliases
    if arg == "off":
        arg = "none"
    elif arg == "on":
        arg = "low"

    try:
        prov.reasoning_effort = arg  # type: ignore[attr-defined]
    except ValueError as e:
        await update.message.reply_text(f"{e}")
        return
    await update.message.reply_text(f"reasoning_effort -> {prov.reasoning_effort}")  # type: ignore[attr-defined]


async def _cmd_reboot(update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorized(update):
        return
    await update.message.reply_text("⟲ rebooting Krypton... back in a moment.")
    asyncio.get_event_loop().call_later(0.6, _exec_self)


def _exec_self() -> None:
    """Spawn a fresh detached process, then hard-exit. Avoids os.execv path-quoting bug on Windows."""
    import os
    import subprocess
    import sys

    log.info("rebooting Krypton: spawning fresh detached process")
    log_path = settings.data_dir / "logs" / "telegram.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logf = open(log_path, "ab")

    flags = 0
    if sys.platform == "win32":
        flags = 0x00000008 | 0x00000200 | 0x01000000  # DETACHED|NEW_PG|BREAKAWAY

    subprocess.Popen(
        [sys.executable, "-m", "krypton", "telegram"],
        cwd=str(settings.workdir),
        stdin=subprocess.DEVNULL,
        stdout=logf,
        stderr=logf,
        creationflags=flags,
        close_fds=False,
    )
    os._exit(0)


# ---------------------------------------------------------------------------
# Inbound handlers — text, document, photo, voice
# ---------------------------------------------------------------------------


def _inbox_dir(chat_id: int) -> Path:
    p = settings.workdir / "inbox" / str(chat_id)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _stamp(name_hint: str = "") -> str:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    if name_hint:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name_hint)[:60]
        return f"{ts}_{safe}"
    return ts


async def _on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorized(update):
        return
    msg = update.message
    if not msg or not msg.text:
        return
    await _process_user_input(update, ctx, user_text=msg.text.strip())


async def _on_document(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorized(update):
        return
    msg = update.message
    if not msg or not msg.document:
        return
    chat_id = update.effective_chat.id
    doc = msg.document
    fname = _stamp(doc.file_name or "file")
    target = _inbox_dir(chat_id) / fname
    tg_file = await doc.get_file()
    await tg_file.download_to_drive(custom_path=str(target))
    caption = (msg.caption or "").strip()
    descr = (
        f"[user sent a file]\n"
        f"  saved to: {target}\n"
        f"  original name: {doc.file_name}\n"
        f"  mime: {doc.mime_type}\n"
        f"  size: {doc.file_size} bytes"
    )
    if caption:
        descr += f"\n  caption: {caption}"
    await _process_user_input(update, ctx, user_text=descr)


async def _on_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorized(update):
        return
    msg = update.message
    if not msg or not msg.photo:
        return
    chat_id = update.effective_chat.id
    photo = msg.photo[-1]  # highest-res variant
    target = _inbox_dir(chat_id) / f"{_stamp('photo')}.jpg"
    tg_file = await photo.get_file()
    await tg_file.download_to_drive(custom_path=str(target))
    caption = (msg.caption or "").strip()
    descr = (
        f"[user sent a photo]\n"
        f"  saved to: {target}\n"
        f"  size: {photo.file_size} bytes  ({photo.width}x{photo.height})"
    )
    if caption:
        descr += f"\n  caption: {caption}"
    await _process_user_input(update, ctx, user_text=descr)


async def _on_voice(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorized(update):
        return
    msg = update.message
    voice = msg.voice if msg else None
    if not voice:
        return
    chat_id = update.effective_chat.id
    target = _inbox_dir(chat_id) / f"{_stamp('voice')}.ogg"
    tg_file = await voice.get_file()
    await tg_file.download_to_drive(custom_path=str(target))
    descr = (
        f"[user sent a voice note]\n"
        f"  saved to: {target}\n"
        f"  duration: {voice.duration}s"
    )
    # Telegram Premium auto-transcription, if any
    transcription = getattr(msg, "voice_transcription", None)
    if transcription:
        descr += f"\n  transcription: {transcription}"
    await _process_user_input(update, ctx, user_text=descr)


# ---------------------------------------------------------------------------
# Core turn handler — shared by text/document/photo/voice
# ---------------------------------------------------------------------------


async def _process_user_input(
    update: Update,
    _ctx: ContextTypes.DEFAULT_TYPE,
    *,
    user_text: str,
) -> None:
    msg = update.message
    chat_id = update.effective_chat.id
    sess = _get_session(chat_id)
    await _ensure_restored(sess, chat_id)

    if sess.lock.locked():
        await msg.reply_text("...still working on the previous request. Use /stop to cancel it.")
        return

    async with sess.lock:
        sess.stop_event.clear()
        await msg.chat.send_action(ChatAction.TYPING)

        # Live status line: edited as tool calls happen so user sees progress.
        status_msg = None
        try:
            status_msg = await msg.reply_text("🤔 thinking…")
        except Exception:  # noqa: BLE001
            pass

        async def _on_tool(name: str, args: dict, res: ToolResult) -> None:
            status = "✓" if res.ok else "✗"
            arg_preview = ", ".join(f"{k}={_short(v)}" for k, v in list(args.items())[:3])
            line = f"{status} {name}({arg_preview}) · {res.duration_ms}ms"
            if status_msg is not None:
                try:
                    await status_msg.edit_text(f"🔧 {line}")
                except (BadRequest, Exception):  # noqa: BLE001
                    pass

        async def _typer():
            try:
                while True:
                    await asyncio.sleep(4)
                    await msg.chat.send_action(ChatAction.TYPING)
            except asyncio.CancelledError:
                return

        typer = asyncio.create_task(_typer())
        sess.current_task = asyncio.create_task(sess.agent.turn(user_text, on_tool=_on_tool))
        try:
            result = await sess.current_task
        except asyncio.CancelledError:
            log.info("turn cancelled for chat %d", chat_id)
            if status_msg is not None:
                try:
                    await status_msg.edit_text("⏹ cancelled.")
                except Exception:  # noqa: BLE001
                    pass
            return
        finally:
            typer.cancel()
            sess.current_task = None

        # Remove the transient status line
        if status_msg is not None:
            try:
                await status_msg.delete()
            except Exception:  # noqa: BLE001
                pass

        # Drain outbox: files the agent decided to send
        while not sess.outbox.empty():
            path = await sess.outbox.get()
            try:
                with open(path, "rb") as f:
                    await msg.reply_document(document=f, filename=path.name)
            except Exception as e:  # noqa: BLE001
                await msg.reply_text(f"failed to send {path.name}: {e}")

        body = result.final_text or "(no answer)"
        if result.aborted and result.error:
            body += f"\n\n⚠️ {result.error}"
        await _send_long(msg, body)

        # Persist updated context
        try:
            await sess.agent.memory.save_conversation(chat_id, sess.agent.context.serialize())
        except Exception as e:  # noqa: BLE001
            log.warning("failed to persist conversation for chat %d: %s", chat_id, e)


# ---------------------------------------------------------------------------
# Output formatting: smart chunking + best-effort MarkdownV2
# ---------------------------------------------------------------------------


_TG_MAX = 4096  # Telegram hard cap per text message
_SAFE_MAX = 3800  # leave margin for formatting overhead


async def _send_long(msg, body: str) -> None:
    """Split a long body smartly and send each chunk.

    Tries MarkdownV2 first (better code-block rendering); on parse failure
    (unbalanced backticks, weird escapes) falls back to plain text. Splits
    at the most natural boundary: code fence > paragraph > line > word > char.
    """
    chunks = _chunk_smart(body, _SAFE_MAX)
    for chunk in chunks:
        sent = False
        md = _to_markdown_v2(chunk)
        if md is not None:
            try:
                await msg.reply_text(md, parse_mode=ParseMode.MARKDOWN_V2, disable_web_page_preview=True)
                sent = True
            except BadRequest as e:
                log.debug("markdown parse failed (%s), falling back to plain text", e)
            except Exception as e:  # noqa: BLE001
                log.debug("send failed (%s); retrying as plain", e)
        if not sent:
            await msg.reply_text(chunk, disable_web_page_preview=True)


def _chunk_smart(s: str, n: int) -> list[str]:
    """Split `s` into chunks of <= n chars.

    Prefers natural boundaries (in order): end of code fence > blank line >
    newline > whitespace > hard cut. Preserves code-fence balance across
    chunks by closing an open fence at the end of a chunk and reopening it
    (with the original language hint) at the start of the next.
    """
    if len(s) <= n:
        return [s]

    chunks: list[str] = []
    remaining = s
    carry_lang: str | None = None  # if not None, we're inside an unclosed code block

    while True:
        prefix = f"```{carry_lang}\n" if carry_lang is not None else ""
        if len(prefix) + len(remaining) <= n:
            chunks.append(prefix + remaining)
            return chunks

        budget = n - len(prefix) - 6  # leave headroom for a possible "\n```" closer
        if budget < 100:
            budget = n - len(prefix)  # pathological: tiny budget
        window = remaining[:budget]
        split = _pick_split(window, budget)
        piece_orig = remaining[:split]
        remaining = remaining[split:]

        # Compute fence state at end of this piece, accounting for the prefix.
        in_code = carry_lang is not None
        lang_buf = carry_lang or ""
        for m in re.finditer(r"(?m)^```([^\n]*)$", piece_orig):
            if not in_code:
                in_code = True
                lang_buf = m.group(1).strip()
            else:
                in_code = False
                lang_buf = ""

        piece = prefix + piece_orig
        if in_code:
            piece = piece.rstrip("\n") + "\n```"
            carry_lang = lang_buf
        else:
            carry_lang = None

        chunks.append(piece)


def _pick_split(window: str, budget: int) -> int:
    """Find the best position to split `window` in [budget//2, budget]."""
    min_pos = max(1, budget // 2)
    # 1) end of a code-fence line (prefer splitting AFTER a complete fence line)
    best = -1
    for m in re.finditer(r"(?m)^```[^\n]*$", window):
        end = m.end()
        # include the trailing newline if present
        if end < len(window) and window[end] == "\n":
            end += 1
        if min_pos <= end <= budget:
            best = end
    if best > 0:
        return best
    # 2) blank line
    p = window.rfind("\n\n")
    if p >= min_pos:
        return p + 2
    # 3) any newline
    p = window.rfind("\n")
    if p >= min_pos:
        return p + 1
    # 4) whitespace
    p = window.rfind(" ")
    if p >= min_pos:
        return p + 1
    # 5) hard cut
    return budget


_MDV2_SPECIAL = r"_*[]()~`>#+-=|{}.!\\"


def _to_markdown_v2(text: str) -> str | None:
    """Convert plain text with ``` fences into MarkdownV2.

    Escapes outside code spans, leaves code untouched. Returns None if the
    text is empty or obviously unsafe (we'd rather plain-send than risk it).
    """
    if not text:
        return None

    out: list[str] = []
    parts = re.split(r"(```[^`]*```|`[^`\n]*`)", text, flags=re.DOTALL)
    for part in parts:
        if not part:
            continue
        if part.startswith("```"):
            # block code: escape only the closing backslashes/backticks risk
            out.append(part)
        elif part.startswith("`") and part.endswith("`"):
            out.append(part)
        else:
            # escape all MarkdownV2 specials
            out.append(_mdv2_escape(part))
    return "".join(out)


def _mdv2_escape(s: str) -> str:
    return re.sub(r"([_*\[\]()~`>#+\-=|{}.!\\])", r"\\\1", s)


def _short(v: Any) -> str:
    s = str(v)
    return s if len(s) < 40 else s[:37] + "..."


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def build_application() -> Application:
    if not settings.telegram_bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set in .env")
    app = ApplicationBuilder().token(settings.telegram_bot_token).build()
    app.add_handler(CommandHandler("start", _cmd_start))
    app.add_handler(CommandHandler("help", _cmd_start))
    app.add_handler(CommandHandler("reset", _cmd_reset))
    app.add_handler(CommandHandler("stop", _cmd_stop))
    app.add_handler(CommandHandler("provider", _cmd_provider))
    app.add_handler(CommandHandler("model", _cmd_model))
    app.add_handler(CommandHandler("status", _cmd_status))
    app.add_handler(CommandHandler("reasoning", _cmd_reasoning))
    app.add_handler(CommandHandler("thinking", _cmd_reasoning))  # alias
    app.add_handler(CommandHandler("reboot", _cmd_reboot))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _on_text))
    app.add_handler(MessageHandler(filters.Document.ALL, _on_document))
    app.add_handler(MessageHandler(filters.PHOTO, _on_photo))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, _on_voice))
    return app


def run_telegram() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = build_application()
    if not settings.authorized_telegram_ids:
        log.warning(
            "⚠️  TELEGRAM_AUTHORIZED_USER_IDS is empty — the bot will accept commands from ANY user. "
            "Set the variable in .env to restrict access."
        )
    else:
        log.info("Krypton bot polling — authorized IDs: %s", settings.authorized_telegram_ids)
    app.run_polling(close_loop=False, allowed_updates=Update.ALL_TYPES)
