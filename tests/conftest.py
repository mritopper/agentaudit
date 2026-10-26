import logging

import pytest

from agentaudit.graph import Auditor
from agentaudit.judges import HeuristicJudge
from agentaudit.schemas import Step, Trace
from agentaudit.store import Store

logging.getLogger("agentaudit").setLevel(logging.ERROR)


@pytest.fixture(autouse=True)
def _no_ambient_credentials(monkeypatch, tmp_path):
    """Tests must never pick up a developer's real Claude credentials."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("AGENTAUDIT_JUDGE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))


@pytest.fixture
def store():
    return Store(":memory:")


@pytest.fixture
def auditor(store):
    return Auditor(HeuristicJudge(), store)


def call(tool, **args):
    return Step(type="tool_call", tool=tool, args=args)


def result(content, error=False):
    return Step(type="tool_result", content=content, error=error)


def make_trace(task="What is order 4412's status?", steps=None, answer="Order 4412 has shipped.", **kw):
    kw.setdefault("trace_id", "t-1")
    kw.setdefault("agent", "support-bot")
    return Trace(task=task, steps=steps or [], final_answer=answer, **kw)
