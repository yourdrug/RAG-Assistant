"""CLI-команда: запуск Arq worker для обработки фоновых задач."""

from __future__ import annotations

import logging
import sys
from collections.abc import Sequence

from arq.connections import RedisSettings
from arq.cron import cron
from arq.worker import Function, Worker, func as arq_func

from config import settings
from infrastructure.worker.tasks import (
    cron_bm25_rebuild,
    cron_job_cleanup,
    cron_recover_orphaned_jobs,
    cron_recover_stuck_processing,
    process_document,
    run_benchmark,
    run_full_ingest,
    run_single_ingest,
    run_sweep,
)

logger = logging.getLogger("cli")

_PROCESS_TIMEOUT = 60 * 30  # single document incl. OCR: 30 min
_SINGLE_INGEST_TIMEOUT = 60 * 30  # single file ingest: 30 min
_FULL_INGEST_TIMEOUT = 60 * 120  # whole corpus ingest: 2 h
_BENCHMARK_TIMEOUT = 60 * 60  # benchmark run: 1 h
_SWEEP_TIMEOUT = 60 * 60 * 6  # parameter sweep with LLM judge: 6 h
_CRON_TIMEOUT = 60 * 10  # maintenance cron jobs: 10 min


def worker(
        max_jobs: int | None = None,
        health_check_interval: int = 10,
) -> None:
    """Запустить Arq worker для обработки фоновых задач.

    Слушает очереди: document_processing, ingest, benchmark.
    Cron-задачи: job_cleanup (2x/day), recover_orphaned (15 мин), bm25_rebuild (3:00 UTC).
    Использует Redis как брокер.
    """
    try:
        redis_settings = RedisSettings.from_dsn(settings.redis_url)

        functions: Sequence[Function] = [
            arq_func(process_document, timeout=_PROCESS_TIMEOUT, keep_result=0),
            arq_func(run_full_ingest, timeout=_FULL_INGEST_TIMEOUT, keep_result=0),
            arq_func(run_single_ingest, timeout=_SINGLE_INGEST_TIMEOUT, keep_result=0),
            arq_func(run_benchmark, timeout=_BENCHMARK_TIMEOUT, keep_result=0),
            arq_func(run_sweep, timeout=_SWEEP_TIMEOUT, keep_result=0),
            arq_func(cron_job_cleanup, timeout=_CRON_TIMEOUT, keep_result=0),
            arq_func(cron_recover_orphaned_jobs, timeout=_CRON_TIMEOUT, keep_result=0),
            arq_func(cron_recover_stuck_processing, timeout=_CRON_TIMEOUT, keep_result=0),
            arq_func(cron_bm25_rebuild, timeout=_CRON_TIMEOUT, keep_result=0),
        ]

        cron_jobs = [
            cron(cron_job_cleanup, hour={1, 13}, timeout=_CRON_TIMEOUT),
            cron(cron_recover_orphaned_jobs, minute={0, 15, 30, 45}, timeout=_CRON_TIMEOUT),
            cron(cron_recover_stuck_processing, minute={5, 20, 35, 50}, timeout=_CRON_TIMEOUT),
            cron(cron_bm25_rebuild, hour=3, minute=0, timeout=_CRON_TIMEOUT),
        ]

        if max_jobs is None:
            max_jobs = settings.worker_max_concurrent

        w = Worker(
            functions=functions,
            cron_jobs=cron_jobs,
            redis_settings=redis_settings,
            max_jobs=max_jobs,
            health_check_interval=health_check_interval,
            queue_name="document_processing",
            on_startup=_on_startup,
            on_shutdown=_on_shutdown,
        )

        logger.info(
            "Arq worker starting — queues: document_processing, ingest, benchmark "
            "cron: cleanup/recover/reconcile/bm25 max_jobs=%d redis=%s",
            max_jobs,
            settings.redis_host,
        )
        w.run()
    except ImportError:
        logger.error("arq package not installed. Run: pip install arq redis")
        sys.exit(1)
    except Exception as exc:
        logger.error("Worker startup failed", exc_info=exc)
        sys.exit(1)


async def _on_startup(ctx: dict) -> None:
    """Initialize database and infrastructure on worker startup."""
    from composition.container import Container
    from infrastructure.database.database import database
    from infrastructure.initialization import _seed_domain_config_defaults
    from infrastructure.redis.redis_client import redis_client

    await database.connect()
    logger.info("Worker: database connected")

    await redis_client.init()
    logger.info("Worker: Redis connected")

    # Build DI container (same as API process)
    container = Container()
    container.init(database)
    ctx["container"] = container

    # Seed domain config defaults (same as the API lifespan). The worker can
    # start before the API has ever seeded (parallel compose boot / worker-only
    # deployment) — without this, legal/general profiles KeyError on their
    # config parameters (e.g. fingerprint_min_articles) during first ingestion.
    domain_registry = container.infrastructure.domain_registry
    if domain_registry is not None and container.infrastructure.uow_factory is not None:
        await _seed_domain_config_defaults(container.infrastructure.uow_factory, domain_registry)

    listener = container.infrastructure.config_listener
    if listener is None:
        raise RuntimeError("Config listener failed to initialize in worker")

    await listener.resync(trigger="worker_startup")
    logger.info("Worker: config synced from database")

    await listener.start()
    ctx["config_listener"] = listener
    logger.info("Worker: config listener started")


async def _on_shutdown(ctx: dict) -> None:
    """Cleanup on worker shutdown."""
    from infrastructure.database.database import database
    from infrastructure.redis.redis_client import redis_client

    listener = ctx.get("config_listener")
    if listener:
        await listener.stop()
        logger.info("Worker: config listener stopped")

    await redis_client.aclose()
    logger.info("Worker: Redis disconnected")

    await database.disconnect()
    logger.info("Worker: database disconnected")
