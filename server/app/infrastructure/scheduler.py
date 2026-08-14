"""APScheduler-based periodic job system.

Wraps ``AsyncIOScheduler`` and exposes ``start_scheduler`` / ``stop_scheduler``
lifespan helpers plus a ``schedule_periodic`` decorator for registering
recurring background tasks (e.g. infra-metrics collection).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import wraps
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sqlalchemy import text

logger = logging.getLogger("default")


def handle_exceptions(func: Callable) -> Callable:
    """Decorator that logs exceptions without crashing the scheduler."""

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await func(*args, **kwargs)
        except Exception:
            logger.exception("Scheduler job %s failed", func.__name__)

    return wrapper


class Scheduler:
    """AsyncIO scheduler for periodic background jobs."""

    def __init__(self) -> None:
        self.scheduler = AsyncIOScheduler(timezone="UTC")

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
        # Periodic job cleanup (every hour)
        self.add_interval_job(
            job_id="periodic_job_cleanup",
            func=self._periodic_job_cleanup,
            seconds=3600,
        )

        # Infrastructure metrics collector (every 30s)
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

        # Recovery orphaned jobs (every 5 min)
        self.add_interval_job(
            job_id="recover_orphaned_jobs",
            func=self._periodic_recover_orphaned_jobs,
            seconds=300,
        )

    @staticmethod
    @handle_exceptions
    async def _periodic_job_cleanup() -> None:
        """Periodically delete old background job records."""
        from presentation.api.dependencies import _uow_factory  # nested to avoid circular import

        async with _uow_factory.create(master=True) as uow:
            from config import settings  # nested to avoid circular import

            deleted = await uow.background_jobs.delete_old(days=settings.job_cleanup_days)
            if deleted:
                logger.info("Cleaned up %d old background jobs", deleted)

    @staticmethod
    @handle_exceptions
    async def _periodic_infra_collector() -> None:
        """Periodically update infrastructure Prometheus gauges."""
        from infrastructure.ml.metrics import collect_infra_metrics  # nested to avoid circular import

        await collect_infra_metrics()

    @staticmethod
    @handle_exceptions
    async def _periodic_config_resync() -> None:
        """Страховочная полная сверка config_parameters -> settings.

        Не заменяет LISTEN/NOTIFY (тот даёт near-real-time применение), а закрывает
        редкое окно потери NOTIFY между network blip и переустановкой listener.
        """
        from presentation.api.dependencies import get_config_listener  # nested to avoid circular import

        listener = get_config_listener()
        if listener.is_connected:
            await listener.resync(trigger="periodic")

    @staticmethod
    @handle_exceptions
    async def _periodic_recover_orphaned_jobs() -> None:
        """Recover background jobs stuck in 'running' state.

        When a worker dies (OOM, deploy, crash), its jobs stay in 'running'
        forever.  This job marks them as 'failed' after a configurable timeout.
        """
        from presentation.api.dependencies import _uow_factory  # nested to avoid circular import

        # Jobs stuck in 'running' for more than 15 minutes are considered orphaned
        orphan_timeout_minutes = 15

        async with _uow_factory.create(master=True) as uow:
            result = await uow._session.execute(
                text(
                    """
                    UPDATE background_jobs
                    SET status = 'failed',
                        error_message = 'Worker died or restarted — task orphaned',
                        finished_at = NOW()
                    WHERE status = 'running'
                      AND started_at < NOW() - make_interval(mins => :timeout)
                    RETURNING id
                    """
                ),
                {"timeout": f"{orphan_timeout_minutes} minutes"},
            )
            orphaned_ids = [row[0] for row in result.fetchall()]
            if orphaned_ids:
                logger.warning("Recovered %d orphaned jobs: %s", len(orphaned_ids), orphaned_ids)
            orphaned_ids = [row[0] for row in result.fetchall()]
            if orphaned_ids:
                logger.warning("Recovered %d orphaned jobs: %s", len(orphaned_ids), orphaned_ids)

    async def startup(self) -> None:
        """Configure and start the scheduler."""
        self._configure()
        self.scheduler.start()
        logger.info("Scheduler started.")

    async def shutdown(self) -> None:
        """Stop the scheduler gracefully."""
        self.scheduler.remove_all_jobs()
        self.scheduler.shutdown(wait=True)
        logger.info("Scheduler stopped.")


scheduler = Scheduler()
