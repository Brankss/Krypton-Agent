"""Hub-and-spoke orchestration: spawn specialist worker subagents in parallel.

The main (orchestrator) agent calls `spawn_agents` with a list of INDEPENDENT
subtasks. Each becomes a fresh worker Agent with:
  * its own scoped tool registry (work tools only — no orchestration / chat tools),
  * its own isolated conversation context,
  * a specialist role prompt and a short iteration budget,
  * the SHARED provider + memory (so we don't fan out N model clients / DB handles).

Workers run concurrently via asyncio.gather (bounded by a semaphore); only their
final results flow back to the orchestrator, keeping its context clean. Workers
get no `spawn_agents` tool, so depth is capped at 1 — no fork bombs.
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable

from krypton.config import settings
from krypton.core.agent import Agent
from krypton.core.context import ConversationContext
from krypton.memory.store import MemoryStore
from krypton.providers.base import LLMProvider
from krypton.tools.base import BaseTool, ToolResult
from krypton.tools.registry import ToolRegistry

# Keep the orchestrator's context readable: cap how much each worker returns.
_PER_WORKER_CHARS = 6000


class SpawnAgentsTool(BaseTool):
    name = "spawn_agents"
    description = (
        "Delegate INDEPENDENT subtasks to specialist worker agents that run IN PARALLEL, "
        "each in its own isolated context with its own tools, then collect their results. "
        "Use it when a task splits into parts that don't depend on each other — research "
        "several topics at once, analyze multiple datasets/competitors, draft several "
        "assets. Give each worker a `role` and SELF-CONTAINED `instructions` (a worker "
        "cannot see this conversation). Do NOT use it for trivial or strictly-sequential "
        "work — do that yourself. Workers cannot spawn further agents."
    )
    parameters = {
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "minItems": 1,
                "description": "Independent subtasks to run in parallel (one worker each).",
                "items": {
                    "type": "object",
                    "properties": {
                        "role": {
                            "type": "string",
                            "description": "Short specialist label, e.g. 'researcher', 'data_analyst', 'copywriter'.",
                        },
                        "instructions": {
                            "type": "string",
                            "description": "Self-contained task for this worker (no reference to the current chat).",
                        },
                    },
                    "required": ["role", "instructions"],
                },
            }
        },
        "required": ["tasks"],
    }
    timeout_s = 1800.0  # workers do real work; each is bounded by its own iteration cap

    def __init__(
        self,
        *,
        get_provider: Callable[[], LLMProvider],
        memory: MemoryStore,
        make_worker_registry: Callable[[], ToolRegistry],
        max_parallel: int | None = None,
        max_workers: int | None = None,
        worker_iterations: int | None = None,
    ) -> None:
        self._get_provider = get_provider
        self._memory = memory
        self._make_worker_registry = make_worker_registry
        self._max_parallel = max_parallel or settings.subagent_max_parallel
        self._max_workers = max_workers or settings.subagent_max_workers
        self._worker_iterations = worker_iterations or settings.subagent_max_iterations

    async def run(self, tasks: list[dict[str, Any]]) -> ToolResult:
        if not isinstance(tasks, list) or not tasks:
            return ToolResult.failure("tasks must be a non-empty list of {role, instructions}")

        specs: list[tuple[str, str]] = []
        for t in tasks:
            if not isinstance(t, dict):
                continue
            role = (str(t.get("role") or "worker").strip()[:60]) or "worker"
            instr = str(t.get("instructions") or "").strip()
            if instr:
                specs.append((role, instr))
        if not specs:
            return ToolResult.failure("no valid tasks (each needs non-empty instructions)")

        truncated = len(specs) > self._max_workers
        specs = specs[: self._max_workers]

        provider = self._get_provider()
        sem = asyncio.Semaphore(max(1, self._max_parallel))

        async def _one(idx: int, role: str, instr: str) -> tuple[int, str, ToolResult]:
            async with sem:
                sub = Agent(
                    provider=provider,                      # shared (httpx is concurrency-safe)
                    registry=self._make_worker_registry(),  # fresh, scoped, no spawn tool
                    memory=self._memory,                    # shared memory
                    context=ConversationContext(target_budget=settings.context_budget),
                    interface="subagent",
                    subagent_role=role,
                    max_iterations=self._worker_iterations,
                )
                res = await sub.turn(instr)
                return idx, role, res

        outcomes = await asyncio.gather(
            *[_one(i, role, instr) for i, (role, instr) in enumerate(specs)],
            return_exceptions=True,
        )

        rows: list[tuple[int, str, str]] = []
        artifacts: list = []
        ok_count = 0
        for oc in outcomes:
            if isinstance(oc, BaseException):
                rows.append((1_000_000, "worker", f"✗ crashed: {type(oc).__name__}: {oc}"))
                continue
            idx, role, res = oc
            if res.aborted and res.error:
                body = f"✗ error: {res.error}"
            else:
                ok_count += 1
                body = (res.final_text or "(no output)").strip()
                if len(body) > _PER_WORKER_CHARS:
                    body = body[:_PER_WORKER_CHARS] + f"\n... [truncated {len(body) - _PER_WORKER_CHARS} chars]"
            if res.artifacts:
                artifacts.extend(res.artifacts)
            rows.append((idx, role, body))

        rows.sort(key=lambda r: r[0])
        out = [f"Spawned {len(specs)} worker(s) in parallel — {ok_count}/{len(specs)} succeeded:\n"]
        for _idx, role, body in rows:
            out.append(f"### [{role}]\n{body}\n")
        if truncated:
            out.append(f"(note: only the first {self._max_workers} tasks were run)")

        return ToolResult(
            ok=ok_count > 0,
            content="\n".join(out),
            artifacts=artifacts,
            meta={"workers": len(specs), "succeeded": ok_count},
        )
