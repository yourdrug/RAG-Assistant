"""BackgroundJob Repository interface."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol


class BackgroundJob:
    __slots__ = (
        "id",
        "job_type",
        "status",
        "related_id",
        "request_id",
        "started_at",
        "finished_at",
        "error_message",
        "creation_date",
    )

    def __init__(
        self,
        id: int | None = None,
        job_type: str = "",
        status: str = "pending",
        related_id: int | None = None,
        request_id: str | None = None,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
        error_message: str | None = None,
        creation_date: datetime | None = None,
    ) -> None:
        self.id = id
        self.job_type = job_type
        self.status = status
        self.related_id = related_id
        self.request_id = request_id
        self.started_at = started_at
        self.finished_at = finished_at
        self.error_message = error_message
        self.creation_date = creation_date


class BackgroundJobRepository(Protocol):
    async def create(self, job: BackgroundJob) -> BackgroundJob: ...
    async def mark_running(self, job_id: int) -> None: ...
    async def mark_done(self, job_id: int) -> None: ...
    async def mark_failed(self, job_id: int, error: str) -> None: ...
    async def count_active(self) -> int: ...
    async def delete_old(self, days: int) -> int: ...
    async def list_recent(self, limit: int = 50, offset: int = 0) -> list[BackgroundJob]: ...
    async def get_by_id(self, job_id: int) -> BackgroundJob | None: ...
    async def count_by_status(self) -> dict[str, int]: ...
