"""Background job lifecycle status."""

from __future__ import annotations

from enum import StrEnum

from domain.shared.enums import LifecycleStatusMixin


class BackgroundJobStatus(
    LifecycleStatusMixin,
    StrEnum,
    terminal=frozenset({"done", "failed"}),
    active=frozenset({"pending", "running"}),
):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
