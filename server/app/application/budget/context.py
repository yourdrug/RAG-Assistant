"""Explicit identity for paid calls, propagated by request/job entry points."""

from contextvars import ContextVar


billing_principal: ContextVar[str | None] = ContextVar("billing_principal", default=None)
