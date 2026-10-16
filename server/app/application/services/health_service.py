"""Application service for system health checks."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import logging
from typing import TYPE_CHECKING

from application.ports.health import HealthCheckResult
from domain.value_objects.health_status import HealthStatus
from domain.value_objects.ml_provider import MLProvider


if TYPE_CHECKING:
    from application.admin import CorpusControlPort, ProviderConfigurationPort
    from application.ports.health import ConfigListenerProviderPort, HealthProbePort
    from application.ports.health_settings import HealthSettingsPort
    from application.ports.unit_of_work_factory import UnitOfWorkFactory


logger = logging.getLogger("default")


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
        *,
        configuration: ProviderConfigurationPort | None = None,
        corpus_control: CorpusControlPort | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._probe = probe
        self._config_listener = config_listener_provider
        self._settings = health_settings
        self._configuration = configuration
        self.corpus_control = corpus_control

    async def check(self) -> HealthResponse:
        qdrant, postgres, redis = await asyncio.gather(
            self._probe.check_qdrant(), self._probe.check_postgres(), self._probe.check_redis()
        )
        ml_checks = (
            await self._probe.check_tei() if self._settings.ml_provider == MLProvider.TEI.value else {}
        )
        active_jobs = await self.count_active_jobs()

        llm_provider = self._settings.llm_provider

        if llm_provider == "openrouter":
            llm_check = await self._probe.check_openrouter()
        else:
            llm_check = await self._probe.check_ollama()

        config_listener_status = HealthCheckResult(
            status=HealthStatus.OK.value if self._config_listener.is_connected else "error: not connected"
        )

        configuration_status = HealthCheckResult(
            status=HealthStatus.OK.value
            if self._configuration is None or self._configuration.status().ready
            else "error: configuration snapshot not applied"
        )
        corpus_status = HealthCheckResult(
            status=HealthStatus.OK.value
            if self.corpus_control is None or await self.corpus_control.is_ready()
            else "error: corpus generation not ready"
        )
        overall = HealthStatus.HEALTHY.value
        error_checks = (
            qdrant,
            llm_check,
            postgres,
            redis,
            config_listener_status,
            configuration_status,
            corpus_status,
            *ml_checks.values(),
        )

        if any(c.status.startswith("error") for c in error_checks):
            overall = HealthStatus.DEGRADED.value

        return HealthResponse(
            status=overall,
            version=self._settings.version,
            uptime_seconds=self._settings.uptime_seconds,
            llm_provider=llm_provider,
            checks={
                **ml_checks,
                "api": HealthCheckResult(status=HealthStatus.OK.value),
                "qdrant": qdrant,
                "llm": llm_check,
                "postgres": postgres,
                "redis": redis,
                "config_listener": config_listener_status,
                "configuration": configuration_status,
                "corpus": corpus_status,
            },
            background_jobs={"running": active_jobs},
        )

    async def count_active_jobs(self) -> int:
        try:
            async with self._uow_factory.create() as uow:
                return await uow.background_jobs.count_active()
        except Exception:
            logger.exception("Failed to count active background jobs during health check")
            return 0
