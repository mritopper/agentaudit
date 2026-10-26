"""SQLite persistence for audit results and the human review queue."""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime
from pathlib import Path

from .schemas import AuditResult, ReviewState, utcnow

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audits (
    trace_id TEXT PRIMARY KEY,
    agent TEXT NOT NULL,
    verdict TEXT NOT NULL,
    composite INTEGER NOT NULL,
    needs_review INTEGER NOT NULL,
    review_status TEXT,
    trace_started_at TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audits_agent ON audits(agent);
CREATE INDEX IF NOT EXISTS idx_audits_started ON audits(trace_started_at);
"""


class Store:
    def __init__(self, path: str | Path = "data/agentaudit.db"):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._db.executescript(_SCHEMA)

    def save(self, result: AuditResult) -> None:
        """Insert or replace. Re-auditing a trace keeps a human decision already made on it."""
        with self._lock:
            row = self._db.execute("SELECT payload FROM audits WHERE trace_id = ?", (result.trace_id,)).fetchone()
            if row:
                previous = AuditResult.model_validate_json(row["payload"])
                if previous.review and previous.review.status != "pending":
                    result.review = previous.review
            self._db.execute(
                "INSERT OR REPLACE INTO audits VALUES (?,?,?,?,?,?,?,?)",
                (
                    result.trace_id, result.agent, result.verdict, result.composite, int(result.needs_review),
                    result.review.status if result.review else None,
                    result.trace_started_at.isoformat(), result.model_dump_json(),
                ),
            )
            self._db.commit()

    def get(self, trace_id: str) -> AuditResult | None:
        with self._lock:
            row = self._db.execute("SELECT payload FROM audits WHERE trace_id = ?", (trace_id,)).fetchone()
        return AuditResult.model_validate_json(row["payload"]) if row else None

    def list(
        self,
        agent: str | None = None,
        verdict: str | None = None,
        review_status: str | None = None,
        since: datetime | None = None,
        limit: int = 500,
    ) -> list[AuditResult]:
        clauses, params = [], []
        if since:
            clauses.append("trace_started_at >= ?"); params.append(since.isoformat())
        if agent:
            clauses.append("agent = ?"); params.append(agent)
        if verdict:
            clauses.append("verdict = ?"); params.append(verdict)
        if review_status:
            clauses.append("review_status = ?"); params.append(review_status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._db.execute(
                f"SELECT payload FROM audits {where} ORDER BY trace_started_at DESC LIMIT ?", (*params, limit)
            ).fetchall()
        return [AuditResult.model_validate_json(r["payload"]) for r in rows]

    def agents(self) -> list[str]:
        with self._lock:
            rows = self._db.execute("SELECT DISTINCT agent FROM audits ORDER BY agent").fetchall()
        return [r[0] for r in rows]

    def review(self, trace_id: str, decision: str, note: str = "") -> AuditResult | None:
        """Record a human decision on a flagged run. decision: 'approved' or 'rejected'."""
        result = self.get(trace_id)
        if result is None or result.review is None:
            return None
        result.review = ReviewState(
            status=decision, reasons=result.review.reasons, note=note, reviewed_at=utcnow()
        )
        with self._lock:
            self._db.execute(
                "UPDATE audits SET review_status = ?, payload = ? WHERE trace_id = ?",
                (decision, result.model_dump_json(), trace_id),
            )
            self._db.commit()
        return result

    def clear(self) -> None:
        with self._lock:
            self._db.execute("DELETE FROM audits")
            self._db.commit()
