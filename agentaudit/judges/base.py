"""Judge interface: anything that can score one dimension of one trace."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..schemas import AuditorReport, Dimension, Signals, Trace


@dataclass
class JudgeOutput:
    report: AuditorReport
    tokens_in: int = 0
    tokens_out: int = 0
    note: str = ""


class JudgeError(Exception):
    """The judge could not produce a usable report (refusal, API failure, bad output)."""


class Judge(Protocol):
    name: str

    def audit(self, dimension: Dimension, trace: Trace, signals: Signals) -> JudgeOutput: ...
