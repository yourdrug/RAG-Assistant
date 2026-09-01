"""Health check ports — abstract interfaces for system health probes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class HealthCheckResult:
    status: str
    latency_ms: float | None = None
    models: list[str] | None = None


@runtime_checkable
class HealthProbePort(Protocol):
    async def check_ollama(self) -> HealthCheckResult: ...
    async def check_openrouter(self) -> HealthCheckResult: ...
    async def check_deepinfra(self) -> HealthCheckResult: ...
    def check_qdrant(self) -> HealthCheckResult: ...
    async def check_postgres(self) -> HealthCheckResult: ...


@runtime_checkable
class ConfigListenerProviderPort(Protocol):
    @property
    def is_connected(self) -> bool: ...
