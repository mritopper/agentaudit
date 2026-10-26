"""HTTP API and dashboard host.

No authentication: this is a local/dev service. Put it behind your own auth
before exposing it, since trace bodies can contain sensitive data.
"""

from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import __version__
from .analytics import summarize
from .graph import Auditor
from .judges import claude_available, get_judge
from .schemas import AuditResult, Trace, utcnow
from .seed import sample_traces
from .store import Store

STATIC = Path(__file__).parent / "static"


class ReviewRequest(BaseModel):
    decision: Literal["approved", "rejected"]
    note: str = ""


def _brief(r: AuditResult) -> dict:
    """List-view projection: everything except the full trace and per-auditor detail."""
    return {
        "trace_id": r.trace_id, "agent": r.agent, "version": r.version, "model": r.model, "task": r.task,
        "started_at": r.trace_started_at, "verdict": r.verdict, "composite": r.composite, "summary": r.summary,
        "dimensions": {d.dimension.value: d.score for d in r.dimensions},
        "modes": sorted({f.mode.value for f in r.findings if f.severity.value != "low"}),
        "needs_review": r.needs_review, "review_status": r.review.status if r.review else None,
        "judge": r.telemetry.judge,
    }


def create_app(db_path: str | None = None, judge_mode: str | None = None) -> FastAPI:
    load_dotenv()
    app = FastAPI(title="AgentAudit", version=__version__)
    store = Store(db_path or os.environ.get("AGENTAUDIT_DB", "data/agentaudit.db"))
    default_mode = judge_mode or os.environ.get("AGENTAUDIT_JUDGE", "auto")
    auditors: dict[str, Auditor] = {}

    def auditor_for(mode: str | None) -> Auditor:
        mode = (mode or default_mode).lower()
        if mode == "claude" and not claude_available():
            raise HTTPException(400, "Claude judge requested but no credentials found. Set ANTHROPIC_API_KEY.")
        judge = get_judge(mode)
        if judge.name not in auditors:
            auditors[judge.name] = Auditor(judge, store)
        return auditors[judge.name]

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/health")
    def health():
        judge = get_judge(default_mode)
        return {
            "status": "ok", "version": __version__, "judge": judge.name,
            "model": getattr(judge, "model", None), "claude_available": claude_available(),
        }

    def _since(days: int | None):
        return utcnow() - timedelta(days=days) if days else None

    @app.get("/api/agents")
    def agents():
        return store.agents()

    @app.get("/api/summary")
    def summary(agent: str | None = None, days: int | None = None):
        return summarize(store.list(agent=agent, since=_since(days), limit=5000))

    @app.get("/api/audits")
    def audits(
        agent: str | None = None, verdict: str | None = None, review_status: str | None = None,
        days: int | None = None, limit: int = 500,
    ):
        return [_brief(r) for r in store.list(agent, verdict, review_status, _since(days), min(limit, 5000))]

    @app.get("/api/audits/{trace_id}")
    def audit_detail(trace_id: str) -> AuditResult:
        result = store.get(trace_id)
        if result is None:
            raise HTTPException(404, f"No audit for trace {trace_id!r}")
        return result

    @app.post("/api/traces")
    def submit(body: Trace | list[Trace], judge: str | None = None):
        traces = body if isinstance(body, list) else [body]
        return [_brief(r) for r in auditor_for(judge).audit_many(traces)]

    @app.post("/api/audits/{trace_id}/review")
    def review(trace_id: str, body: ReviewRequest) -> AuditResult:
        result = store.review(trace_id, body.decision, body.note)
        if result is None:
            raise HTTPException(404, f"Trace {trace_id!r} not found or was not flagged for review")
        return result

    @app.post("/api/seed")
    def seed(judge: str = "heuristic"):
        # Defaults to the free offline judge so loading demo data never spends API credit by surprise.
        traces = sample_traces()
        return {"audited": len(auditor_for(judge).audit_many(traces)), "judge": judge}

    return app
