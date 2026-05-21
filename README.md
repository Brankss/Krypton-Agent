<div align="center">

# Krypton Agent

**A local, autonomous personal AI agent that actually *does* things on your machine.**

Multi-provider LLM • Surgical tool system • Persistent pattern learning • Telegram remote control

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Async](https://img.shields.io/badge/async-asyncio-orange.svg)](https://docs.python.org/3/library/asyncio.html)
[![Providers](https://img.shields.io/badge/providers-Ollama%20%7C%20OpenRouter-purple.svg)](#providers)

</div>

---

## What is Krypton?

Krypton is a **local-first autonomous agent** designed to run on your own hardware and operate on your real system — files, shell, network, Python — with zero confirmation prompts and persistent memory across reboots. You talk to it via a terminal REPL or a private Telegram bot from your phone, and it does the work.

It's not a chatbot, not a playground demo, not a wrapper around someone else's framework. It's a single ~3k-line Python codebase you can read top-to-bottom in an afternoon, designed to be the **personal assistant that lives on your laptop and gets shit done.**

---

## Why this is different

Most "agent" projects on GitHub are one of three things:
1. A LangChain demo with 7 layers of abstraction and no autonomy.
2. A SaaS wrapper that needs an OpenAI key and ships your data to a third party.
3. A toy that prints tool calls but never actually executes them.

Krypton is none of those. Concretely:

| | Typical agent | **Krypton** |
|---|---|---|
| **Where it runs** | OpenAI / hosted API | Your laptop, Ollama-first, no cloud required |
| **Permission model** | Asks before every action | Permanent grant — executes immediately |
| **Tool dispatch** | Sequential | **Parallel** via `asyncio.gather` |
| **Context management** | Truncate when full → forgets everything | **3-stage compaction**: dedup → head/tail → drop scratch |
| **Memory** | Forgotten on restart | **SQLite-persisted** patterns + facts + full conversations |
| **Learning** | None | `remember_pattern` with Jaccard auto-dedup |
| **Remote access** | Web UI | Private Telegram bot, photos/voice/files inbound |
| **Tool result blow-up** | Floods context → crash | 16 KB cap + middle-elision per tool call |
| **Identical errors spam** | 50 copies of the same line | Fingerprint dedup keeps 1 |
| **Lines of framework** | ~50 000 | ~3 000 |

The design priority is **surgical precision over generality**. Each tool has the narrowest possible scope, the registry caches its schema, the prompt builder caches the static portion, the context manager drops redundancy *before* truncating. Everything is async, everything streams, everything pools its connections.

---

## Feature highlights

- **Three LLM backends, one interface** — Ollama local, Ollama Cloud, OpenRouter. Switch live with `/provider`.
- **19 surgical tools** — filesystem, file CRUD, web search (Tavily + DDG), shell, Python exec, memory.
- **Parallel tool dispatch** — independent calls fire concurrently, not one at a time.
- **Smart context compaction** — identical tool outputs deduped to one entry with a `[repeated N×]` marker; head/tail truncation only as last resort.
- **Persistent memory** — patterns and facts survive restarts; conversations restored per Telegram chat.
- **Pattern learning with auto-dedup** — `remember_pattern` checks Jaccard similarity ≥ 0.55 and merges instead of inserting duplicates.
- **Telegram bot** — private (whitelist by user ID), receives photos/voice/files, live "🔧 tool_name · 142ms" status messages, MarkdownV2 with code-fence-aware chunking.
- **Health-check mode** — `python main.py --doctor` verifies the active provider and tool stack before you commit to a run.
- **Auto-Ollama boot** — if the daemon isn't running when Krypton starts, it spawns `ollama serve` detached and waits for ready.
- **Zero-config default** — copy `.env.example`, fill three values, run `python main.py`.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                          Interfaces                              │
│   ┌────────────┐                              ┌────────────┐    │
│   │  CLI REPL  │                              │  Telegram  │    │
│   │ (rich+pt)  │                              │   bot      │    │
│   └─────┬──────┘                              └─────┬──────┘    │
└─────────┼─────────────────────────────────────────────┼─────────┘
          │              bootstrap.build_agent()        │
          └──────────────────────┬──────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                          Agent loop                              │
│                                                                  │
│   ┌──────────────┐    ┌─────────────────┐    ┌──────────────┐   │
│   │   Context    │◄───┤  Prompt builder │◄───┤   Memory     │   │
│   │   manager    │    │ (static cached) │    │   (SQLite)   │   │
│   └──────┬───────┘    └─────────────────┘    └──────────────┘   │
│          │                                                      │
│          ▼                                                       │
│   ┌──────────────┐    ┌─────────────────┐    ┌──────────────┐   │
│   │   Provider   ├───►│   tool_calls    ├───►│   Registry   │   │
│   │  (streaming) │    │  (parallel)     │    │  (19 tools)  │   │
│   └──────────────┘    └─────────────────┘    └──────┬───────┘   │
│                                                     │           │
└─────────────────────────────────────────────────────┼───────────┘
                                                     ▼
                          ┌────────────────────────────────────────┐
                          │  Filesystem · Shell · Python · Web ·   │
                          │  Memory · Telegram file delivery       │
                          └────────────────────────────────────────┘
```

### Project layout

```
krypton/
├── main.py                # CLI entrypoint (python -m krypton ...)
├── bootstrap.py           # Wires provider + tools + memory + agent
├── config.py              # Pydantic settings from .env
│
├── core/
│   ├── agent.py           # Main loop: stream → tool dispatch (parallel) → repeat
│   ├── context.py         # 3-stage compaction (dedup → truncate → drop scratch)
│   ├── prompt.py          # System prompt: cached static + dynamic per-turn
│   └── tokens.py          # tiktoken-based token estimation
│
├── providers/
│   ├── base.py            # LLMProvider protocol, Message/StreamEvent types
│   ├── ollama.py          # Ollama local + cloud (same wire protocol)
│   ├── openrouter.py      # OpenAI-compatible SSE
│   ├── factory.py         # build_provider(name)
│   └── _msg.py            # Cross-provider message normalization
│
├── tools/
│   ├── base.py            # Tool ABC, ToolResult, timeout wrapper
│   ├── registry.py        # Register/dispatch + cached schema + per-tool stats
│   ├── filesystem.py      # list_directory, find_files, grep
│   ├── files.py           # read/write/edit/append/delete/stat
│   ├── web.py             # web_search, fetch_url (shared httpx pool)
│   ├── shell.py           # execute_python, powershell, shell
│   └── memory.py          # remember/recall/forget_pattern, note/get_fact
│
├── memory/
│   └── store.py           # SQLite WAL — patterns, facts, conversations
│
├── interfaces/
│   ├── cli.py             # REPL with rich + prompt_toolkit
│   └── telegram.py        # Bot + send_file_to_user tool
│
└── runtime/
    └── ollama_boot.py     # Auto-start Ollama daemon if not running
```

### The turn loop, step by step

```
user message
   │
   ▼
1. refresh_system_prompt
   ├─ memory.search_patterns(user_text, top=6)
   ├─ memory.all_facts()
   └─ build_system_prompt(...)  ← static cached, dynamic rebuilt
   │
   ▼
2. context.add_user(text, pinned=True)
   │
   ▼
3. for it in 1..max_iterations:
   │
   ├─ context.compact()          ← dedup → head/tail → drop scratch
   ├─ provider.stream(messages)  ← TextDelta / ToolCallDelta / Done events
   ├─ append assistant message
   │
   ├─ if no tool_calls → return final_text  ✓
   │
   └─ asyncio.gather(dispatch(c) for c in tool_calls)   ← PARALLEL
      └─ append tool_result for each
```

---

## Requirements

| | |
|---|---|
| **Python** | 3.11 or newer |
| **OS** | Windows 11 (primary target) · Linux · macOS |
| **RAM** | 4 GB minimum, 16 GB recommended for local 7B models |
| **GPU (optional)** | Any CUDA / Metal GPU with ≥ 6 GB VRAM for fast local inference |
| **Ollama** | Required for local provider; optional for cloud-only setups |

---

## Setup

### 1 · Clone & install

<details>
<summary><b>Windows (PowerShell)</b></summary>

```powershell
git clone https://github.com/<your-user>/krypton-agent.git
cd krypton-agent

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

</details>

<details>
<summary><b>Linux / macOS (bash/zsh)</b></summary>

```bash
git clone https://github.com/<your-user>/krypton-agent.git
cd krypton-agent

python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

</details>

### 2 · Install Ollama (if using a local model)

| OS | Command |
|---|---|
| **Windows** | Download installer at [ollama.com/download](https://ollama.com/download) |
| **macOS** | `brew install ollama` &nbsp;or installer |
| **Linux** | `curl -fsSL https://ollama.com/install.sh \| sh` |

Then pull a model — the default recommended for an 8 GB GPU:

```bash
ollama pull qwen2.5-coder:7b
```

Other good choices by hardware:

| Hardware | Recommended model | Pull command |
|---|---|---|
| **CPU only / 8 GB RAM** | `qwen2.5-coder:3b` | `ollama pull qwen2.5-coder:3b` |
| **8 GB VRAM GPU** | `qwen2.5-coder:7b` (default) | `ollama pull qwen2.5-coder:7b` |
| **16 GB VRAM GPU** | `qwen2.5-coder:14b` | `ollama pull qwen2.5-coder:14b` |
| **No local hardware** | Use Ollama Cloud or OpenRouter | see below |

Krypton **auto-starts** the Ollama daemon if it's not running — you don't need to manage it manually.

### 3 · Configure `.env`

Copy the template and edit:

```bash
cp .env.example .env
```

Open `.env` and fill in only what you actually use:

```ini
# Choose: ollama_local | ollama_cloud | openrouter
KRYPTON_PROVIDER=ollama_local

# --- Local Ollama (default) ---
OLLAMA_LOCAL_HOST=http://127.0.0.1:11434
OLLAMA_LOCAL_MODEL=qwen2.5-coder:7b

# --- Ollama Cloud (optional, paid) ---
OLLAMA_CLOUD_API_KEY=
OLLAMA_CLOUD_MODEL=gpt-oss:120b

# --- OpenRouter (optional, ~hundreds of models) ---
OPENROUTER_API_KEY=
OPENROUTER_MODEL=anthropic/claude-sonnet-4

# --- Telegram bot (optional, for remote access) ---
TELEGRAM_BOT_TOKEN=
TELEGRAM_AUTHORIZED_USER_IDS=

# --- Web search (optional, falls back to free DuckDuckGo) ---
TAVILY_API_KEY=

# --- Runtime paths (adjust to your install) ---
KRYPTON_WORKDIR=/path/to/krypton-agent
KRYPTON_DATA_DIR=/path/to/krypton-agent/data
```

### 4 · Verify

```bash
python main.py --doctor
```

If you see `provider OK · N tools registered` you're done.

---

## Usage

### Launcher (`main.py`)

```bash
python main.py             # foreground Telegram bot (Ctrl+C to stop)
python main.py --detach    # background bot — survives terminal close
python main.py --stop      # kill any running instance
python main.py --cli       # local terminal REPL instead of Telegram
python main.py --doctor    # health-check the provider
python main.py --no-kill   # skip auto-killing existing instances
```

The launcher automatically kills stale instances before starting a new one, so you'll never have two bots fighting over the same Telegram poll.

### CLI REPL

```bash
python main.py --cli
```

Typing commands inside the REPL:

| Command | Effect |
|---|---|
| `/provider <ollama_local\|ollama_cloud\|openrouter>` | Switch backend live |
| `/model <name>` | Switch model (e.g. `/model qwen2.5-coder:14b`) |
| `/reset` | Wipe the conversation context |
| `/patterns` | List learned patterns |
| `/status` | Show provider, model, tool call stats |
| `/quit` | Exit |

### Telegram bot

```bash
python main.py             # foreground
python main.py --detach    # background
```

In Telegram chat with your bot:

| Command | Effect |
|---|---|
| `/start` | Greet and confirm authorization |
| `/reset` | Wipe the conversation history for this chat |
| `/status` | Provider, model, message count, last 5 tools used |
| `/provider <name>` | Switch backend |
| `/model <name>` | Switch model |
| `/stop` | Cancel the currently-running turn |
| `/reboot` | Restart the bot process (in-place) |

The bot accepts **text, photos, voice notes, documents** — files are saved under `inbox/<chat_id>/` and the agent is told where to find them. The agent can send files back with the `send_file_to_user` tool.

---

## Tool inventory

| # | Tool | Purpose | Notes |
|---|---|---|---|
| 1 | `list_directory` | Compact directory listing | Skips `node_modules`, `.git`, `__pycache__` |
| 2 | `find_files` | Recursive glob | Uses `scandir`, prunes junk dirs |
| 3 | `grep` | Multi-file regex search | Bounded concurrency (semaphore=64), early-stop, binary skip |
| 4 | `read_file` | Read with offset/limit | Streaming — no full-file load for slices |
| 5 | `write_file` | Atomic write | tmp + rename |
| 6 | `edit_file` | Exact-string replacement | Fails if the match isn't unique |
| 7 | `append_file` | Append to file | |
| 8 | `delete_path` | Delete file or dir | |
| 9 | `stat_path` | File metadata | |
| 10 | `web_search` | Tavily or DuckDuckGo | 5-min cache, snippet cap 240 chars |
| 11 | `fetch_url` | Fetch + HTML→text | Shared `httpx` pool |
| 12 | `execute_python` | Run Python subprocess | `-I` isolated mode, configurable timeout |
| 13 | `powershell` | PowerShell `-NoProfile -NonInteractive` | Windows only |
| 14 | `shell` | `cmd.exe` / `sh -c` | |
| 15 | `remember_pattern` | Store a learned recipe / error / optimization | Jaccard auto-dedup ≥ 0.55 |
| 16 | `recall_patterns` | Keyword search over patterns | |
| 17 | `forget_pattern` | Delete stale patterns | |
| 18 | `note_fact` / `get_fact` | Key/value durable facts | Auto-injected into every prompt |
| 19 | `send_file_to_user` | Send a file via Telegram | Only registered when running the bot |

---

## How the memory system works

Krypton's SQLite database (`data/krypton.db`) has three tables:

### Patterns
Free-text lessons the agent stores via `remember_pattern`. Each pattern has a `kind` (`recipe`, `known_error`, `optimization`, `preference`), tags, and a body. At the start of every turn the agent runs `search_patterns(user_text, limit=6)` and the top hits are injected into the system prompt — so past experience automatically informs future actions without you having to do anything.

When the agent calls `remember_pattern` with content similar to an existing one (Jaccard ≥ 0.55 on word tokens), the new content is merged in rather than duplicated.

### Facts
Key/value pairs like `preferred_editor=vscode`, `gh_user=riccardo123`. Set with `note_fact`, fetched with `get_fact`, and the full set is injected into every system prompt.

### Conversations
For the Telegram bot, the full per-chat context is serialized to SQLite after each turn and restored lazily on the next message. This means restarting the bot or rebooting your machine doesn't lose the conversation — the agent picks up exactly where it left off, with full context.

---

## Context management

Naive context handling means: "messages fill the window → truncate the oldest → agent forgets what you told it 3 minutes ago."

Krypton's `ConversationContext.compact()` runs a 3-stage pipeline on every turn:

1. **Dedup pass** — compute a fingerprint for each entry. If 50 tool calls returned the same error, all but the most recent are dropped and the survivor is tagged `[repeated 50×]`. Pair-safety passes ensure no `tool_call_id` is left dangling without its response (or vice versa).
2. **Head/tail truncation** — if still over budget, keep the pinned system prompt + the most recent N entries + the very first user turn (for context).
3. **Drop scratch** — non-pinned scratch entries (intermediate tool outputs the agent already consumed) get dropped last.

Tool results larger than 16 KB are middle-elided before they ever enter the context: head + `... [N chars elided] ...` + tail. The agent is told it can re-run with stricter limits if it actually needs the elided content.

---

## Providers in detail

### `ollama_local`
- Talks to a local Ollama daemon at `OLLAMA_LOCAL_HOST` (default `http://127.0.0.1:11434`).
- Auto-starts the daemon if not running.
- Sends `keep_alive: "30m"` to keep the model warm in VRAM between turns.
- Zero cost, zero latency to a third party, full privacy.

### `ollama_cloud`
- Same wire protocol as local — just a different host + `Authorization: Bearer <key>`.
- Access to large models you can't run locally (`gpt-oss:120b`, etc).
- Some models require a paid subscription; `qwen3-coder-next:cloud` is free as of writing.

### `openrouter`
- OpenAI-compatible SSE.
- Hundreds of models, pay-per-token: Claude, GPT-4o, Gemini, Llama, Mistral, etc.
- Best when you want the absolute top-end models without managing keys for each provider.

Switch live in REPL or Telegram with `/provider <name>` and `/model <name>`. The agent rebuilds its provider object instantly without losing context.

---

## Telegram bot setup walkthrough

1. **Create the bot.** Talk to [@BotFather](https://t.me/BotFather) on Telegram → `/newbot` → follow the prompts. Copy the token it gives you.

2. **Find your user ID.** Talk to [@userinfobot](https://t.me/userinfobot) → it replies with your numeric ID. Copy that number.

3. **Fill `.env`:**
   ```ini
   TELEGRAM_BOT_TOKEN=1234567890:ABCdef...
   TELEGRAM_AUTHORIZED_USER_IDS=123456789
   ```
   Multiple IDs go comma-separated. Anyone not whitelisted is silently ignored.

4. **Start it:**
   ```bash
   python main.py --detach
   ```

5. **Open Telegram → your bot → `/start`.** You're online.

To stop it from anywhere: `python main.py --stop`. To restart it remotely without a terminal: send `/reboot` to the bot.

---

## The autonomy model

The agent's system prompt explicitly disallows confirmation-seeking. The forbidden phrases ("Vuoi che proceda?", "Posso fare X?", etc.) are listed by name in the operating directives. When the user message looks like an imperative ("fai", "manda", "esegui", ...), a dynamic `# This turn` block is appended telling the model it MUST emit tool calls in this turn.

The one exception: genuinely destructive or irreversible operations (mass delete, schema migration, data-loss risk) — the agent is instructed to propose a 3-bullet plan and ask once. Everything else: act.

This is intentional and at the express request of the user. If you fork this, you may want to tune the autonomy directives in `krypton/core/prompt.py` to your own taste.

---

## Performance choices

- **Static prompt cached** by tool-registry signature — no re-stringifying 19 tool descriptions every turn.
- **Tool schema cached** in the registry, invalidated on register.
- **Tool dispatch in parallel** via `asyncio.gather` when calls are independent.
- **Shared `httpx.AsyncClient`** with connection pooling for web tools.
- **Streaming `read_file` and `grep`** — never load whole files into memory.
- **Binary file detection** via NUL-byte sniff in `grep` — skips them before regex.
- **Token estimation** with `tiktoken` (cl100k_base) — fast and accurate enough.
- **SQLite WAL mode** for concurrent reads while the bot is writing.
- **Pattern search** in two stages: SQL `LIKE` for candidates (capped at 100), Python Jaccard rerank for relevance.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `provider error: HTTP 403` on an Ollama Cloud model | That model requires a paid subscription. Switch to `qwen3-coder-next:cloud` (free) or use `ollama_local`. |
| Bot replies are truncated mid-code-block | Already handled by `_chunk_smart`; if it still happens, file an issue with the exact message length. |
| "tool_call_id orphaned" provider error | Run `python main.py --stop` then start fresh. Pair-safety should prevent this; report if reproducible. |
| `ollama serve` won't start on Windows | Make sure no other process is on port 11434. The auto-boot waits 60 s; if your machine is slow, bump `ready_timeout` in `runtime/ollama_boot.py`. |
| Telegram bot silently ignores you | Your user ID isn't in `TELEGRAM_AUTHORIZED_USER_IDS`. Check with `@userinfobot`. |
| `pip install -e .` fails on pydantic | Wipe and reinstall: `rm -rf .venv && python -m venv .venv && pip install -e .` |
| Agent says "ok procedo" but doesn't act | Re-read your `.env` — make sure you're on a model that supports function calling. `qwen2.5-coder:7b` and above do; older / smaller models often don't. |

---

## Roadmap / ideas

- [ ] Voice transcription for incoming Telegram voice notes (whisper.cpp)
- [ ] OCR for incoming photos (tesseract)
- [ ] Background scheduled tasks (cron-style)
- [ ] Browser automation tool (Playwright)
- [ ] Multi-user mode with per-user memory partitions
- [ ] Self-update via `git pull` + restart

PRs welcome.

---

## Contributing

This is a personal project published as open source — issues and PRs are welcome but I make no commitment to maintain it as a community project. If you fork it for your own setup, that's the intended use.

```bash
git clone https://github.com/<your-user>/krypton-agent.git
cd krypton-agent
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
ruff check krypton/
pytest
```

---

## License

[MIT](LICENSE) — do whatever you want, just don't sue me.

---

## Credits

- Built with [Ollama](https://ollama.com), [python-telegram-bot](https://python-telegram-bot.org), [httpx](https://www.python-httpx.org), [pydantic](https://docs.pydantic.dev), [rich](https://rich.readthedocs.io), [prompt_toolkit](https://python-prompt-toolkit.readthedocs.io).
- Authored by Riccardo with assistance from [Claude](https://claude.com).
