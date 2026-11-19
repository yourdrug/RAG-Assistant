"""Health check schemas."""

from __future__ import annotations

from pydantic import BaseModel


class HealthCheck(BaseModel):
    status: str
    latency_ms: float | None = None
    models: list[str] | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float
    checks: dict[str, HealthCheck]
    background_jobs: dict[str, int]


class IndexingQueueResponse(BaseModel):
    pending: int
    running: int
    oldest_ready_seconds: int | None
    expired_leases: int


class IndexingHealthResponse(BaseModel):
    queue: IndexingQueueResponse
    degraded: bool


class OperationalHealthResponse(BaseModel):
    storage: HealthCheck
    indexing: IndexingHealthResponse | None
