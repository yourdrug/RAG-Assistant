"""BackgroundJob entity — tracking for async background tasks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

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
