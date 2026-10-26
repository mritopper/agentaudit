"""Data models shared by the audit pipeline, the store and the API."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- taxonomy


class Dimension(str, Enum):
    """The four quality dimensions, each owned by one auditor node."""

    correctness = "correctness"
    grounding = "grounding"
    tool_use = "tool_use"
    safety = "safety"


class Severity(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class FailureMode(str, Enum):
    """Closed failure taxonomy: every finding maps to exactly one mode, so the
    dashboard can count and trend them."""

    task_failure = "task_failure"
    hallucination = "hallucination"
    ignored_tool_error = "ignored_tool_error"
    redundant_loop = "redundant_loop"
    tool_misuse = "tool_misuse"
    pii_leak = "pii_leak"
    prompt_injection = "prompt_injection"
    policy_violation = "policy_violation"


SEVERITY_RANK = {Severity.low: 0, Severity.medium: 1, Severity.high: 2, Severity.critical: 3}


# --------------------------------------------------------------------------- input: a trace


class Step(BaseModel):
    """One event in an agent run."""

    type: Literal["thought", "tool_call", "tool_result", "message"]
    content: str = ""
    tool: str | None = None
    args: dict[str, Any] | None = None
    error: bool = False
    latency_ms: int | None = None


class Trace(BaseModel):
    """A single agent run: what was asked, what the agent did, what it answered."""

    trace_id: str
    agent: str
    version: str = "v1"
    model: str = "unknown"
    task: str
    steps: list[Step] = Field(default_factory=list)
    final_answer: str = ""
    started_at: datetime = Field(default_factory=utcnow)
    cost_usd: float | None = None
    tokens: int | None = None
    # Governance context: what the agent is allowed to do.
    available_tools: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    pii_allowed: bool = False

    @field_validator("started_at")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        # Naive timestamps are taken as UTC so every comparison downstream is aware-vs-aware.
        return v.replace(tzinfo=timezone.utc) if v.tzinfo is None else v.astimezone(timezone.utc)


# --------------------------------------------------------------------------- deterministic signals


class PIIHit(BaseModel):
    kind: str
    masked: str
    in_task: bool


class Signals(BaseModel):
    """Facts computed by plain code before any LLM sees the trace. Judges receive
    these as evidence; the heuristic judge is built entirely on them."""

    n_steps: int = 0
    n_tool_calls: int = 0
    n_tool_errors: int = 0
    tool_error_rate: float = 0.0
    max_repeat: int = 1
    repeated_calls: list[str] = Field(default_factory=list)
    unknown_tools: list[str] = Field(default_factory=list)
    forbidden_calls: list[int] = Field(default_factory=list)
    unresolved_error_steps: list[int] = Field(default_factory=list)
    error_acknowledged: bool = False
    pii_in_answer: list[PIIHit] = Field(default_factory=list)
    injection_steps: list[int] = Field(default_factory=list)
    injection_followed_steps: list[int] = Field(default_factory=list)
    ungrounded_claims: list[str] = Field(default_factory=list)
    answer_relevance: float = 1.0
    total_latency_ms: int = 0


# --------------------------------------------------------------------------- auditor output


class Finding(BaseModel):
    mode: FailureMode
    severity: Severity
    summary: str
    evidence_steps: list[int]
    recommendation: str


class AuditorReport(BaseModel):
    """What one auditor returns. This exact model is the JSON schema Claude is
    constrained to, so keep every field required."""

    score: int = Field(description="0-100, where 100 means no problems on this dimension")
    confidence: int = Field(description="0-100, how sure the auditor is of its own score")
    rationale: str
    findings: list[Finding]


class DimensionReport(AuditorReport):
    dimension: Dimension
    judge: str
    latency_ms: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    degraded: bool = False
    note: str = ""
    shadow_score: int | None = None  # the free rule-based score, kept to cross-check an LLM judge


# --------------------------------------------------------------------------- final result


class TaggedFinding(Finding):
    dimension: Dimension


class ReviewState(BaseModel):
    status: Literal["pending", "approved", "rejected"] = "pending"
    reasons: list[str] = Field(default_factory=list)
    note: str = ""
    reviewed_at: datetime | None = None


class Telemetry(BaseModel):
    """Health of the auditor itself: the QA system needs QA too."""

    judge: str
    total_ms: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    degraded_dimensions: list[str] = Field(default_factory=list)


class AuditResult(BaseModel):
    trace_id: str
    agent: str
    version: str
    model: str
    task: str
    trace_started_at: datetime
    audited_at: datetime = Field(default_factory=utcnow)
    composite: int
    verdict: Literal["pass", "warn", "fail"]
    summary: str
    dimensions: list[DimensionReport]
    findings: list[TaggedFinding]
    signals: Signals
    needs_review: bool
    review: ReviewState | None = None
    telemetry: Telemetry
    trace: Trace
