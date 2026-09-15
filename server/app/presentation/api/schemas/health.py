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
    llm_provider: str
    background_jobs: dict[str, int]
