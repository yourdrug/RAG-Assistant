"""APScheduler-based periodic job system.

Wraps ``AsyncIOScheduler`` and exposes ``start_scheduler`` / ``stop_scheduler``
lifespan helpers plus a ``schedule_periodic`` decorator for registering
recurring background tasks (e.g. infra-metrics collection).

Note: heavy maintenance jobs (cleanup, orphan recovery)
are handled by Arq worker cron — see infrastructure.worker.cron.
Only process-local jobs stay here (metrics collector, config resync).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from infrastructure.metrics.metrics import collect_infra_metrics
from infrastructure.utils.async_exception_handler import handle_exceptions


if TYPE_CHECKING:
    from collections.abc import Callable

    from application.ports.ml_clients import MLClientPort
    from infrastructure.events.postgres_config_listener import PostgresConfigListener


logger = logging.getLogger("default")


class Scheduler:
    """AsyncIO scheduler for periodic background jobs."""

    def __init__(self) -> None:
        self.scheduler = AsyncIOScheduler(timezone="UTC")
        self._config_listener: PostgresConfigListener | None = None
        self._ml_clients: MLClientPort | None = None

    def add_interval_job(
        self,
        job_id: str,
        func: Callable[[], object],
        seconds: int,
    ) -> None:
        self.scheduler.add_job(
            id=job_id,
            func=func,
            replace_existing=True,
            trigger="interval",
            max_instances=1,
            seconds=seconds,
        )

    def configure(self) -> None:
        # Infrastructure metrics collector (every 30s) — server-local Prometheus registry
        self.add_interval_job(
            job_id="infra_metrics_collector",
            func=self.periodic_infra_collector,
            seconds=30,
        )

        # Config resync — страховка от потерянных NOTIFY (every 5 min)
        self.add_interval_job(
            job_id="config_resync",
            func=self.periodic_config_resync,
            seconds=300,
        )

    @handle_exceptions
    async def periodic_infra_collector(self) -> None:
        """Periodically update infrastructure Prometheus gauges."""
        await collect_infra_metrics(ml_clients=self._ml_clients)

    @handle_exceptions
    async def periodic_config_resync(self) -> None:
        """Страховочная полная сверка config_parameters -> settings.

        Не заменяет LISTEN/NOTIFY (тот даёт near-real-time применение), а закрывает
        редкое окно потери NOTIFY между network blip и переустановкой listener.
        """
        if self._config_listener is not None and self._config_listener.is_connected:
            await self._config_listener.resync(trigger="periodic")

    async def startup(
        self,
        config_listener: PostgresConfigListener | None = None,
        ml_clients: MLClientPort | None = None,
    ) -> None:
        """Configure and start the scheduler with required dependencies."""
        self._config_listener = config_listener
        self._ml_clients = ml_clients
        self.configure()
        self.scheduler.start()
        logger.info("Scheduler started.")

    async def shutdown(self) -> None:
        """Stop the scheduler gracefully."""
        self.scheduler.remove_all_jobs()
        self.scheduler.shutdown(wait=True)
        logger.info("Scheduler stopped.")


scheduler = Scheduler()
