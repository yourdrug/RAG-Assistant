"""APScheduler-based periodic job system.

Wraps ``AsyncIOScheduler`` and exposes ``start_scheduler`` / ``stop_scheduler``
lifespan helpers plus a ``schedule_periodic`` decorator for registering
recurring background tasks (e.g. infra-metrics collection).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler

if TYPE_CHECKING:
    from application.ports.unit_of_work_factory import UnitOfWorkFactory

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
        self._uow_factory: UnitOfWorkFactory | None = None
        self._config_listener: Any = None
        self._ml_clients: Any = None

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

    def add_cron_job(
        self,
        job_id: str,
        func: Callable,
        hour: int,
        minute: int = 0,
        **kwargs: Any,
    ) -> None:
        self.scheduler.add_job(
            id=job_id,
            func=func,
            replace_existing=True,
            trigger="cron",
            max_instances=1,
            hour=hour,
            minute=minute,
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

        # BM25 index full rebuild (daily at 3:00 AM UTC)
        self.add_cron_job(
            job_id="bm25_daily_rebuild",
            func=self._periodic_bm25_rebuild,
            hour=3,
            minute=0,
        )

    @handle_exceptions
    async def _periodic_job_cleanup(self) -> None:
        """Periodically delete old background job records."""
        from config import settings

        async with self._uow_factory.create(master=True) as uow:
            deleted = await uow.background_jobs.delete_old(days=settings.job_cleanup_days)
            if deleted:
                logger.info("Cleaned up %d old background jobs", deleted)

    @handle_exceptions
    async def _periodic_infra_collector(self) -> None:
        """Periodically update infrastructure Prometheus gauges."""
        from infrastructure.ml.metrics import collect_infra_metrics

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
    async def _periodic_recover_orphaned_jobs(self) -> None:
        """Recover background jobs stuck in 'running' state."""
        async with self._uow_factory.create(master=True) as uow:
            orphaned_ids = await uow.background_jobs.recover_orphaned(timeout_minutes=15)
            if orphaned_ids:
                logger.warning("Recovered %d orphaned jobs: %s", len(orphaned_ids), orphaned_ids)

    @handle_exceptions
    async def _periodic_bm25_rebuild(self) -> None:
        """Rebuild BM25 index from scratch daily."""
        import time
        from pathlib import Path

        from config import settings

        from infrastructure.ml.hybrid import BM25Index, save_bm25_index

        if not settings.hybrid_enabled:
            logger.debug("BM25 rebuild skipped: hybrid search disabled")
            return

        t0 = time.monotonic()

        async with self._uow_factory.create(master=True) as uow:
            all_texts = await uow.chunks.get_all_contents()

        if not all_texts:
            logger.info("BM25 rebuild: no chunks found, skipping")
            return

        bm25_index = BM25Index(all_texts)
        bm25_path = Path(settings.data_dir) / "bm25_index.json"
        save_bm25_index(bm25_index, bm25_path)

        # BM25 cache is now managed by MLClientRegistry — no manual clear needed

        elapsed = time.monotonic() - t0
        logger.info(
            "BM25 daily rebuild completed: %d chunks indexed in %.1fs",
            len(all_texts),
            elapsed,
        )

    async def startup(
        self,
        uow_factory: UnitOfWorkFactory,
        config_listener: Any = None,
        ml_clients: Any = None,
    ) -> None:
        """Configure and start the scheduler with required dependencies."""
        self._uow_factory = uow_factory
        self._config_listener = config_listener
        self._ml_clients = ml_clients
        self._configure()
        self.scheduler.start()
        logger.info("Scheduler started.")

    async def shutdown(self) -> None:
        """Stop the scheduler gracefully."""
        self.scheduler.remove_all_jobs()
        self.scheduler.shutdown(wait=True)
        logger.info("Scheduler stopped.")


scheduler = Scheduler()
