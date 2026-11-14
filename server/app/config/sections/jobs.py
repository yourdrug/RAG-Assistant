"""Background worker, outbox and job maintenance settings."""

from __future__ import annotations

from pydantic import BaseModel, PositiveInt

from config.ownership import Static


class JobsSection(BaseModel):
    # Broker slots per worker replica (capacity = replicas x slots).
    worker_max_concurrent: Static[PositiveInt] = 4  # batch role
    maintenance_max_concurrent: Static[PositiveInt] = 2
    document_max_concurrent: Static[PositiveInt] = 4  # documents role
    job_max_per_principal: Static[PositiveInt] = 2
    job_backlog_limit: Static[PositiveInt] = 5000
    job_principal_backlog_limit: Static[PositiveInt] = 1000
    worker_completion_wait_seconds: Static[PositiveInt] = 90
    # Cluster-wide caps on running jobs, independent of replica count (enforced at claim).
    ingest_max_concurrent: Static[PositiveInt] = 1  # full and single-file ingest together
    benchmark_job_max_concurrent: Static[PositiveInt] = 1
    sweep_max_concurrent: Static[PositiveInt] = 1
    outbox_max_concurrent: Static[PositiveInt] = 5
    job_cleanup_days: PositiveInt = 30
    stuck_job_timeout_minutes: int = 30
    stale_pending_timeout_minutes: int = 30
