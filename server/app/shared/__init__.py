"""Shared context variables -- used by both infrastructure and presentation.

Lives outside any layer to avoid circular dependencies.
"""

from __future__ import annotations

from contextvars import ContextVar

request_id_ctx: ContextVar[str] = ContextVar("request_id", default="")
