"""Background job lifecycle status."""

from __future__ import annotations

from enum import StrEnum


class BackgroundJobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
