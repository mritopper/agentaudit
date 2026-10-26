"""Judge selection: Claude when credentials exist, otherwise the offline heuristic judge."""

from __future__ import annotations

import os
from pathlib import Path

from .base import Judge, JudgeError, JudgeOutput
from .heuristic import HeuristicJudge

__all__ = ["Judge", "JudgeError", "JudgeOutput", "HeuristicJudge", "get_judge", "claude_available"]


def claude_available() -> bool:
    """True if the SDK can find credentials: an API key, an auth token or an `ant auth login` profile."""
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    return (Path.home() / ".config" / "anthropic").is_dir()


def get_judge(mode: str | None = None) -> Judge:
    """mode: 'claude', 'heuristic' or 'auto' (default; from AGENTAUDIT_JUDGE)."""
    mode = (mode or os.environ.get("AGENTAUDIT_JUDGE", "auto")).lower()
    if mode == "heuristic":
        return HeuristicJudge()
    if mode in ("claude", "auto"):
        if mode == "auto" and not claude_available():
            return HeuristicJudge()
        from .claude import DEFAULT_MODEL, ClaudeJudge  # imported lazily so offline mode never touches the SDK

        return ClaudeJudge(
            model=os.environ.get("AGENTAUDIT_MODEL", DEFAULT_MODEL),
            effort=os.environ.get("AGENTAUDIT_EFFORT", "medium"),
        )
    raise ValueError(f"Unknown judge mode {mode!r}; use 'auto', 'claude' or 'heuristic'")
