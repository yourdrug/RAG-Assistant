"""Durable background jobs and their lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from domain.shared.enums import LifecycleStatus


class BackgroundJobStatus(
    LifecycleStatus,
    terminal=frozenset({"done", "failed"}),
    active=frozenset({"pending", "running"}),
):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class JobType(StrEnum):
    DOCUMENT_PROCESSING = "document_processing"
    DOCUMENT_DELETION = "document_deletion"
    INGEST = "ingest"
    BENCHMARK = "benchmark"
    SWEEP = "sweep"


@dataclass
class BackgroundJob:
    id: int | None = None
    job_type: str = ""
    status: BackgroundJobStatus = BackgroundJobStatus.PENDING
    related_id: int | None = None
    request_id: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    heartbeat_at: datetime | None = None
    error_message: str | None = None
    creation_date: datetime | None = None

    def __post_init__(self) -> None:
        self.status = BackgroundJobStatus(self.status)

    def mark_running(self) -> None:
        """Transition to RUNNING status."""
        self.status = BackgroundJobStatus.RUNNING
        self.started_at = datetime.now(tz=UTC)
        self.heartbeat_at = self.started_at

    def mark_done(self) -> None:
        """Transition to DONE status."""
        self.status = BackgroundJobStatus.DONE
        self.finished_at = datetime.now(tz=UTC)

    def mark_failed(self, error: str | None = None) -> None:
        """Transition to FAILED status."""
        self.status = BackgroundJobStatus.FAILED
        self.finished_at = datetime.now(tz=UTC)
        self.error_message = error
