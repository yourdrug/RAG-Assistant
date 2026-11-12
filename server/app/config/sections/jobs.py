"""Background worker, outbox and job maintenance settings."""

from __future__ import annotations

from pydantic import BaseModel, PositiveInt

from config.ownership import Static


class JobsSection(BaseModel):
    worker_max_concurrent: Static[PositiveInt] = 4
    outbox_max_concurrent: Static[PositiveInt] = 5
    job_cleanup_days: PositiveInt = 30
    stuck_job_timeout_minutes: int = 30
    stale_pending_timeout_minutes: int = 30
