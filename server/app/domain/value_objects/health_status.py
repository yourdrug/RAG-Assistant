"""Health status constants."""

from __future__ import annotations

from enum import StrEnum


class HealthStatus(StrEnum):
    OK = "ok"
    DEGRADED = "degraded"
    ERROR = "error"
