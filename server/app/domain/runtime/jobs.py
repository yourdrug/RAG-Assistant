"""Durable background jobs and their lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from domain.shared.enums import LifecycleStatus
from domain.shared.exceptions import AppClientError


class BackgroundJobStatus(
    LifecycleStatus,
    terminal=frozenset({"done", "failed", "cancelled"}),
    active=frozenset({"pending", "running"}),
):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(StrEnum):
    DOCUMENT_PROCESSING = "document_processing"
    DOCUMENT_DELETION = "document_deletion"
    INGEST = "ingest"
    INGEST_FILE = "ingest_file"
    BENCHMARK = "benchmark"
    SWEEP = "sweep"


@dataclass(frozen=True)
class ConcurrencyLimit:
    """Cluster-wide cap on simultaneously running jobs of the given types."""

    job_types: frozenset[str]
    max_running: int | None = None
    max_per_principal: int | None = None


class JobAdmissionError(AppClientError):
    """Durable outstanding work has reached an admission limit."""

    def __init__(self, scope: str) -> None:
        self.scope = scope
        self.retry_after_sec = 30
        super().__init__("Background job capacity reached; retry later")


class JobConcurrencyLimitReachedError(Exception):
    """A claimable job must wait: its concurrency group is saturated."""


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
    dispatch_payload: dict[str, object] | None = None

    payload_version: int = 1
    deduplication_key: str | None = None
    attempt: int = 0
    attempt_id: str | None = None
    lease_until: datetime | None = None
    next_attempt_at: datetime | None = None
    dispatch_attempts: int = 0
    progress: dict[str, str] | None = None

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
