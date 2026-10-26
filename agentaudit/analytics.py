"""Roll audit results up into the numbers the dashboard shows.

Computed in Python over stored results; at a few thousand audits this is
instant. Past that, push the group-bys into SQL columns.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date, timedelta
from statistics import mean

from .schemas import AuditResult, Dimension, Severity, utcnow


_SEVERITY_WEIGHT = {Severity.low: 0, Severity.medium: 1, Severity.high: 3, Severity.critical: 5}


def _avg(values) -> float | None:
    values = list(values)
    return round(mean(values), 1) if values else None


def _rolling(by_day: dict[str, list[int]], window: int = 3) -> list[dict]:
    """Trailing mean over `window` days. A fleet with ~1 run per agent per day is too noisy to read raw."""
    out = []
    for day in sorted(by_day):
        end = date.fromisoformat(day)
        scores = [s for d, ss in by_day.items() if 0 <= (end - date.fromisoformat(d)).days < window for s in ss]
        out.append({"date": day, "avg": _avg(scores), "n": len(scores)})
    return out


def _agent_row(agent: str, rows: list[AuditResult]) -> dict:
    now = utcnow()
    recent = [r.composite for r in rows if r.trace_started_at >= now - timedelta(days=7)]
    prior = [r.composite for r in rows if now - timedelta(days=14) <= r.trace_started_at < now - timedelta(days=7)]
    # Rank failure modes by severity-weighted count, so one critical outranks a few mediums.
    modes: Counter = Counter()
    for r in rows:
        for f in r.findings:
            modes[f.mode.value] += _SEVERITY_WEIGHT[f.severity]
    by_version: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        by_version[r.version].append(r.composite)
    return {
        "agent": agent,
        "audits": len(rows),
        "avg_score": _avg(r.composite for r in rows),
        "pass_rate": round(100 * sum(r.verdict == "pass" for r in rows) / len(rows)),
        "open_reviews": sum(1 for r in rows if r.review and r.review.status == "pending"),
        # Change in average score, last 7 days vs the 7 before. None when a window is empty.
        "delta_7d": round(mean(recent) - mean(prior), 1) if recent and prior else None,
        "top_failure": next((m for m, w in modes.most_common(1) if w > 0), None),
        "dimensions": {d.value: _avg(x.score for r in rows for x in r.dimensions if x.dimension == d) for d in Dimension},
        "versions": {v: _avg(s) for v, s in sorted(by_version.items())},
    }


def summarize(results: list[AuditResult]) -> dict:
    if not results:
        return {"kpis": {"audits": 0}, "agents": [], "trend": {}, "failure_modes": [], "auditor": {}}

    by_agent: dict[str, list[AuditResult]] = defaultdict(list)
    for r in results:
        by_agent[r.agent].append(r)

    daily: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        daily[r.agent][r.trace_started_at.date().isoformat()].append(r.composite)
    trend = {agent: _rolling(days) for agent, days in daily.items()}

    findings = [f for r in results for f in r.findings if f.severity != Severity.low]
    mode_rows: dict[str, Counter] = defaultdict(Counter)
    for f in findings:
        mode_rows[f.mode.value][f.severity.value] += 1
    failure_modes = sorted(
        ({"mode": m, "count": sum(c.values()), "critical": c["critical"], "high": c["high"]} for m, c in mode_rows.items()),
        key=lambda x: (-x["count"], x["mode"]),
    )

    n = len(results)
    return {
        "kpis": {
            "audits": n,
            "avg_score": _avg(r.composite for r in results),
            "pass_rate": round(100 * sum(r.verdict == "pass" for r in results) / n),
            "fail_rate": round(100 * sum(r.verdict == "fail" for r in results) / n),
            "open_reviews": sum(1 for r in results if r.review and r.review.status == "pending"),
            "critical_findings": sum(1 for f in findings if f.severity == Severity.critical),
        },
        "agents": sorted((_agent_row(a, rows) for a, rows in by_agent.items()), key=lambda x: x["avg_score"]),
        "trend": trend,
        "failure_modes": failure_modes,
        # Health of the auditor itself: is the QA system working?
        "auditor": {
            "judges": dict(Counter(r.telemetry.judge for r in results)),
            "avg_audit_ms": round(mean(r.telemetry.total_ms for r in results)),
            "tokens_in": sum(r.telemetry.tokens_in for r in results),
            "tokens_out": sum(r.telemetry.tokens_out for r in results),
            "degraded_rate": round(100 * sum(bool(r.telemetry.degraded_dimensions) for r in results) / n),
        },
    }
