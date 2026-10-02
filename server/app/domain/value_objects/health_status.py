"""Health status constants."""

from __future__ import annotations

from enum import StrEnum


class HealthStatus(StrEnum):
    OK = "ok"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    ERROR = "error"
