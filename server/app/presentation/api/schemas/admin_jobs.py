"""Admin background jobs schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class JobResponse(BaseModel):
    id: int
    job_type: str
    status: str
    related_id: int | None = None
    request_id: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error_message: str | None = None
    creation_date: datetime | None = None
    attempt: int = 0
    retryable: bool = False


class JobRetryResponse(BaseModel):
    job_id: int


class JobsListResponse(BaseModel):
    total: int
    jobs: list[JobResponse]


class JobsStatsResponse(BaseModel):
    total: int
    by_status: dict[str, int]
