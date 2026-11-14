"""Initiator of the current request or job; drives per-principal job admission and fairness."""

from contextvars import ContextVar


job_principal: ContextVar[str | None] = ContextVar("job_principal", default=None)
