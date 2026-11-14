"""Execution ownership propagated to transactional application work."""

from contextvars import ContextVar
from dataclasses import dataclass


class ExecutionLeaseLostError(RuntimeError):
    """A superseded attempt must stop and may not commit business changes."""


@dataclass(frozen=True)
class JobExecution:
    job_id: int
    attempt_id: str


current_execution: ContextVar[JobExecution | None] = ContextVar("current_job_execution", default=None)
