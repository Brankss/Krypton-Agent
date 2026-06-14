"""The agent's self-model must match its real cloud environment & role."""
from __future__ import annotations

from krypton.core.prompt import build_system_prompt
from krypton.tools.base import BaseTool, ToolResult
from krypton.tools.registry import ToolRegistry
from krypton.tools.shell import tools as shell_tools


class _Dummy(BaseTool):
    name = "dummy"
    description = "does a thing"
    timeout_s = 5.0

    async def run(self) -> ToolResult:
        return ToolResult.success("")


def _prompt() -> str:
    r = ToolRegistry()
    r.register(_Dummy())
    return build_system_prompt(registry=r, last_user_text="ciao")


def test_knows_it_is_remote_cloud_not_the_users_pc():
    p = _prompt()
    assert "REMOTE CLOUD SERVER" in p
    assert "CANNOT" in p
    # the honest fallback for local files
    assert "inbox/" in p


def test_states_role_breadth_and_scheduling():
    p = _prompt()
    low = p.lower()
    assert "schedule_task" in p
    for kw in ("report", "campaign", "analysis", "personal"):
        assert kw in low


def test_no_windows_powershell_tool_on_cloud():
    names = {t.name for t in shell_tools()}
    assert names == {"execute_python", "shell"}
    assert "powershell" not in names
