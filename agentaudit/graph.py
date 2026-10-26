"""The audit pipeline, expressed as a LangGraph state machine.

    START ─► ingest ─┬─► correctness ─┐
                     ├─► grounding   ─┤
                     ├─► tool_use    ─┼─► adjudicate ─┬─► escalate ─► record ─► END
                     └─► safety      ─┘               └──────────────► record ─► END

`ingest` computes deterministic signals. The four auditors run in parallel, each
scoring one dimension. `adjudicate` merges them into a verdict. Runs that need a
human go through `escalate` (the governance gate) before everything is recorded.
"""

from __future__ import annotations

import logging
import operator
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from .judges import HeuristicJudge, Judge, JudgeError
from .metrics import compute_signals
from .schemas import AuditResult, Dimension, DimensionReport, Signals, Trace
from .scoring import adjudicate
from .store import Store

log = logging.getLogger("agentaudit")


class AuditState(TypedDict, total=False):
    trace: Trace
    signals: Signals
    reports: Annotated[list[DimensionReport], operator.add]  # each parallel auditor appends one
    started: float
    result: AuditResult


class Auditor:
    """Compiles the graph once for a judge and audits traces with it."""

    def __init__(self, judge: Judge, store: Store | None = None, on_escalate: Callable[[AuditResult], None] | None = None):
        self.judge = judge
        self.store = store
        self.on_escalate = on_escalate
        self.graph = self._build()

    # ------------------------------------------------------------------ nodes

    def _ingest(self, state: AuditState) -> dict:
        return {"signals": compute_signals(state["trace"])}

    def _auditor_node(self, dimension: Dimension) -> Callable[[AuditState], dict]:
        fallback = HeuristicJudge()

        def run(state: AuditState) -> dict:
            trace, signals = state["trace"], state["signals"]
            started = time.perf_counter()
            judge_name, degraded, note = self.judge.name, False, ""
            try:
                out = self.judge.audit(dimension, trace, signals)
                note = out.note
            except JudgeError as e:
                # Never lose an audit because a judge failed: degrade to the offline judge and say so.
                log.warning("%s auditor (%s) failed, falling back to heuristic: %s", dimension.value, self.judge.name, e)
                out = fallback.audit(dimension, trace, signals)
                judge_name, degraded, note = fallback.name, True, f"{self.judge.name} judge failed: {e}"
            # Cross-check: when an LLM judges, also run the free rules and keep their score.
            shadow = fallback.audit(dimension, trace, signals).report.score if not degraded and judge_name != fallback.name else None
            report = DimensionReport(
                **out.report.model_dump(),
                dimension=dimension,
                judge=judge_name,
                latency_ms=round((time.perf_counter() - started) * 1000),
                tokens_in=out.tokens_in,
                tokens_out=out.tokens_out,
                degraded=degraded,
                note=note,
                shadow_score=shadow,
            )
            return {"reports": [report]}

        return run

    def _adjudicate(self, state: AuditState) -> dict:
        total_ms = round((time.perf_counter() - state["started"]) * 1000)
        return {"result": adjudicate(state["trace"], state["signals"], state["reports"], total_ms)}

    @staticmethod
    def _route(state: AuditState) -> str:
        return "escalate" if state["result"].needs_review else "record"

    def _escalate(self, state: AuditState) -> dict:
        result = state["result"]
        log.info("review required for %s (%s): %s", result.trace_id, result.agent, "; ".join(result.review.reasons))
        if self.on_escalate:
            self.on_escalate(result)
        return {"result": result}

    def _record(self, state: AuditState) -> dict:
        if self.store:
            self.store.save(state["result"])
        return {"result": state["result"]}

    # ------------------------------------------------------------------ wiring

    def _build(self):
        g = StateGraph(AuditState)
        g.add_node("ingest", self._ingest)
        for dim in Dimension:
            g.add_node(dim.value, self._auditor_node(dim))
            g.add_edge("ingest", dim.value)
        g.add_node("adjudicate", self._adjudicate)
        g.add_edge([d.value for d in Dimension], "adjudicate")  # join: wait for all four auditors
        g.add_node("escalate", self._escalate)
        g.add_node("record", self._record)
        g.add_edge(START, "ingest")
        g.add_conditional_edges("adjudicate", self._route, {"escalate": "escalate", "record": "record"})
        g.add_edge("escalate", "record")
        g.add_edge("record", END)
        return g.compile()

    # ------------------------------------------------------------------ public

    def audit(self, trace: Trace) -> AuditResult:
        state = self.graph.invoke({"trace": trace, "reports": [], "started": time.perf_counter()})
        return state["result"]

    def audit_many(self, traces: list[Trace], max_workers: int = 4) -> list[AuditResult]:
        """Audit traces concurrently; with the Claude judge the time is spent waiting on the API."""
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            return list(pool.map(self.audit, traces))
