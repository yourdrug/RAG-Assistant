"""APScheduler-based periodic job system.

Wraps ``AsyncIOScheduler`` and exposes ``start_scheduler`` / ``stop_scheduler``
lifespan helpers plus a ``schedule_periodic`` decorator for registering
recurring background tasks (e.g. infra-metrics collection).

Note: heavy maintenance jobs (cleanup, orphan recovery, BM25 rebuild)
are handled by Arq worker cron — see infrastructure.worker.tasks.cron_*.
Only process-local jobs stay here (metrics collector, config resync).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import wraps
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from infrastructure.ml.metrics import collect_infra_metrics

logger = logging.getLogger("default")


def handle_exceptions(func: Callable) -> Callable:
    """Log exceptions without crashing the scheduler."""

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.exception(
                "Scheduler job %s failed: [%s] %s",
                func.__name__,
                type(e).__name__,
                e,
            )

    return wrapper


class Scheduler:
    """AsyncIO scheduler for periodic background jobs."""

    def __init__(self) -> None:
        self.scheduler = AsyncIOScheduler(timezone="UTC")
        self._config_listener: Any = None
        self._ml_clients: Any = None
        self._outbox_dispatcher: Any = None

    def add_interval_job(
        self,
        job_id: str,
        func: Callable,
        seconds: int,
        **kwargs: Any,
    ) -> None:
        self.scheduler.add_job(
            id=job_id,
            func=func,
            replace_existing=True,
            trigger="interval",
            max_instances=1,
            seconds=seconds,
            **kwargs,
        )

    def _configure(self) -> None:
        # Infrastructure metrics collector (every 30s) — server-local Prometheus registry
        self.add_interval_job(
            job_id="infra_metrics_collector",
            func=self._periodic_infra_collector,
            seconds=30,
        )

        # Config resync — страховка от потерянных NOTIFY (every 5 min)
        self.add_interval_job(
            job_id="config_resync",
            func=self._periodic_config_resync,
            seconds=300,
        )

        # Outbox dispatch — safety net for missed NOTIFYs (every 30s)
        if self._outbox_dispatcher is not None:
            self.add_interval_job(
                job_id="vector_outbox_dispatch",
                func=self._periodic_outbox_dispatch,
                seconds=30,
            )

            # Reconcile stuck documents (every 60s)
            self.add_interval_job(
                job_id="outbox_reconcile_stuck",
                func=self._periodic_reconcile_stuck,
                seconds=60,
            )

    @handle_exceptions
    async def _periodic_infra_collector(self) -> None:
        """Periodically update infrastructure Prometheus gauges."""
        await collect_infra_metrics(ml_clients=self._ml_clients)

    @handle_exceptions
    async def _periodic_config_resync(self) -> None:
        """Страховочная полная сверка config_parameters -> settings.

        Не заменяет LISTEN/NOTIFY (тот даёт near-real-time применение), а закрывает
        редкое окно потери NOTIFY между network blip и переустановкой listener.
        """
        if self._config_listener is not None and self._config_listener.is_connected:
            await self._config_listener.resync(trigger="periodic")

    @handle_exceptions
    async def _periodic_outbox_dispatch(self) -> None:
        """Safety net: process pending outbox entries if NOTIFY was missed."""
        if self._outbox_dispatcher is None:
            return
        processed = await self._outbox_dispatcher.run_once(batch_size=50)
        if processed:
            logger.info("Outbox dispatch (periodic): processed %d entries", processed)

    @handle_exceptions
    async def _periodic_reconcile_stuck(self) -> None:
        """Fix documents stuck in 'indexing' with no pending outbox entries."""
        if self._outbox_dispatcher is None:
            return
        fixed = await self._outbox_dispatcher.reconcile_stuck_documents()
        if fixed:
            logger.info("Outbox reconcile: fixed %d stuck documents", fixed)

    async def startup(
        self,
        uow_factory: Any = None,
        config_listener: Any = None,
        ml_clients: Any = None,
        outbox_dispatcher: Any = None,
    ) -> None:
        """Configure and start the scheduler with required dependencies."""
        self._config_listener = config_listener
        self._ml_clients = ml_clients
        self._outbox_dispatcher = outbox_dispatcher
        self._configure()
        self.scheduler.start()
        logger.info("Scheduler started.")

    async def shutdown(self) -> None:
        """Stop the scheduler gracefully."""
        self.scheduler.remove_all_jobs()
        self.scheduler.shutdown(wait=True)
        logger.info("Scheduler stopped.")


scheduler = Scheduler()
