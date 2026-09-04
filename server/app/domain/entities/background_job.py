"""BackgroundJob entity — tracking for async background tasks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from domain.value_objects.job_status import BackgroundJobStatus


@dataclass
class BackgroundJob:
    id: int | None = None
    job_type: str = ""
    status: str = BackgroundJobStatus.PENDING.value
    related_id: int | None = None
    request_id: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error_message: str | None = None
    creation_date: datetime | None = None

    def mark_running(self) -> None:
        """Transition to RUNNING status."""
        self.status = BackgroundJobStatus.RUNNING.value
        self.started_at = datetime.now(tz=UTC)

    def mark_done(self) -> None:
        """Transition to DONE status."""
        self.status = BackgroundJobStatus.DONE.value
        self.finished_at = datetime.now(tz=UTC)

    def mark_failed(self, error: str | None = None) -> None:
        """Transition to FAILED status."""
        self.status = BackgroundJobStatus.FAILED.value
        self.finished_at = datetime.now(tz=UTC)
        self.error_message = error
