"""Application service for system health checks."""

from __future__ import annotations

from dataclasses import dataclass, field

from domain.value_objects.health_status import HealthStatus

from application.ports.health import ConfigListenerProviderPort, HealthCheckResult, HealthProbePort
from application.ports.health_settings import HealthSettingsPort
from application.ports.unit_of_work_factory import UnitOfWorkFactory


@dataclass(frozen=True)
class HealthResponse:
    status: str
    version: str
    uptime_seconds: float
    llm_provider: str
    checks: dict[str, HealthCheckResult] = field(default_factory=dict)
    background_jobs: dict[str, int] = field(default_factory=dict)


class HealthService:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        probe: HealthProbePort,
        config_listener_provider: ConfigListenerProviderPort,
        health_settings: HealthSettingsPort,
    ) -> None:
        self._uow_factory = uow_factory
        self._probe = probe
        self._config_listener = config_listener_provider
        self._settings = health_settings

    async def check(self) -> HealthResponse:
        qdrant = self._probe.check_qdrant()
        ollama = await self._probe.check_ollama()
        postgres = await self._probe.check_postgres()
        active_jobs = await self._count_active_jobs()

        config_listener_status = HealthCheckResult(
            status=HealthStatus.OK.value if self._config_listener.is_connected else "error: not connected"
        )

        overall = "healthy"
        if any(c.status.startswith("error") for c in [qdrant, ollama, postgres, config_listener_status]):
            overall = "degraded"

        return HealthResponse(
            status=overall,
            version=self._settings.version,
            uptime_seconds=self._settings.uptime_seconds,
            llm_provider=self._settings.llm_provider,
            checks={
                "api": HealthCheckResult(status=HealthStatus.OK.value),
                "qdrant": qdrant,
                "ollama": ollama,
                "postgres": postgres,
                "config_listener": config_listener_status,
            },
            background_jobs={"running": active_jobs},
        )

    async def _count_active_jobs(self) -> int:
        try:
            async with self._uow_factory.create() as uow:
                return await uow.background_jobs.count_active()
        except Exception:
            return 0
