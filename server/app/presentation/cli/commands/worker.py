"""CLI-команда: запуск Arq worker для обработки фоновых задач."""

from __future__ import annotations

import asyncio
import logging
import sys
from collections.abc import Sequence

from arq.connections import RedisSettings
from arq.cron import cron
from arq.worker import Function, Worker, func as arq_func

from composition.container import Container
from config import settings
from infrastructure.database.database import database
from infrastructure.redis.redis_client import redis_client
from infrastructure.resilience.circuit_breaker import init_breakers
from infrastructure.worker.queue import QUEUE_NAME
from infrastructure.worker.tasks import (
    cron_bm25_rebuild,
    cron_job_cleanup,
    cron_recover_orphaned_jobs,
    cron_recover_stuck_processing,
    process_document,
    run_benchmark,
    run_full_ingest,
    run_single_ingest,
)
from infrastructure.worker.document_deletion import delete_document
from infrastructure.worker.checkpoint_cleanup import cron_sweep_checkpoint_cleanup
from infrastructure.worker.sweep import run_sweep_task as run_sweep

logger = logging.getLogger("cli")

_PROCESS_TIMEOUT = 60 * 30  # single document incl. OCR: 30 min
_SINGLE_INGEST_TIMEOUT = 60 * 30  # single file ingest: 30 min
_FULL_INGEST_TIMEOUT = 60 * 120  # whole corpus ingest: 2 h
_BENCHMARK_TIMEOUT = 60 * 60  # benchmark run: 1 h
_SWEEP_TIMEOUT = 60 * 60 * 6  # parameter sweep with LLM judge: 6 h
_CRON_TIMEOUT = 60 * 10  # maintenance cron jobs: 10 min
_MAX_JOB_TRIES = 3


def worker(
    max_jobs: int | None = None,
    health_check_interval: int = 10,
) -> None:
    """Запустить Arq worker для обработки фоновых задач.

    Слушает очередь: background_tasks.
    Cron-задачи: job_cleanup (2x/day), recover_orphaned (15 мин), bm25_rebuild (3:00 UTC).
    Использует Redis как брокер.
    """
    try:
        redis_settings = RedisSettings.from_dsn(settings.redis_url)

        functions: Sequence[Function] = [
            arq_func(delete_document, timeout=_PROCESS_TIMEOUT, keep_result=0, max_tries=_MAX_JOB_TRIES),
            arq_func(process_document, timeout=_PROCESS_TIMEOUT, keep_result=0, max_tries=10000),
            arq_func(run_full_ingest, timeout=_FULL_INGEST_TIMEOUT, keep_result=0, max_tries=_MAX_JOB_TRIES),
            arq_func(
                run_single_ingest,
                timeout=_SINGLE_INGEST_TIMEOUT,
                keep_result=0,
                max_tries=_MAX_JOB_TRIES,
            ),
            arq_func(run_benchmark, timeout=_BENCHMARK_TIMEOUT, keep_result=0, max_tries=_MAX_JOB_TRIES),
            arq_func(
                run_sweep, name="run_sweep", timeout=_SWEEP_TIMEOUT, keep_result=0, max_tries=_MAX_JOB_TRIES
            ),
            arq_func(cron_sweep_checkpoint_cleanup, timeout=_CRON_TIMEOUT, keep_result=0),
            arq_func(cron_job_cleanup, timeout=_CRON_TIMEOUT, keep_result=0),
            arq_func(cron_recover_orphaned_jobs, timeout=_CRON_TIMEOUT, keep_result=0),
            arq_func(cron_recover_stuck_processing, timeout=_CRON_TIMEOUT, keep_result=0),
            arq_func(cron_bm25_rebuild, timeout=_CRON_TIMEOUT, keep_result=0),
        ]

        cron_jobs = [
            cron(cron_sweep_checkpoint_cleanup, minute=set(range(60)), timeout=_CRON_TIMEOUT),
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
            queue_name=QUEUE_NAME,
            on_startup=on_startup,
            on_shutdown=on_shutdown,
        )

        logger.info(
            "Arq worker starting — queue: %s cron: cleanup/recover/reconcile/bm25 max_jobs=%d redis=%s",
            QUEUE_NAME,
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


async def on_startup(ctx: dict) -> None:
    """Initialize database and infrastructure on worker startup."""
    try:
        await database.connect()
        await redis_client.init()
        container = Container()
        ctx["container"] = container
        container.init(database)
        await container.infrastructure.bootstrap_configuration()
        listener = container.infrastructure.events.config_listener
        if listener is None:
            raise RuntimeError("Config listener must be initialized in worker")
        init_breakers(
            fail_max=settings.llm_breaker_fail_max,
            timeout_duration=settings.llm_breaker_timeout_duration,
        )
        await listener.start()
        ctx["config_listener"] = listener
        ctx["config_resync_task"] = asyncio.create_task(
            periodic_config_resync(listener),
            name="worker-config-resync",
        )
    except BaseException:
        await on_shutdown(ctx)
        raise


async def periodic_config_resync(listener) -> None:
    while True:
        await asyncio.sleep(60)
        await listener.resync(trigger="worker_periodic")


async def on_shutdown(ctx: dict) -> None:
    """Cleanup on worker shutdown."""
    task = ctx.pop("config_resync_task", None)
    if task is not None:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    try:
        container = ctx.pop("container", None)
        if container is not None:
            await container.dispose()
            logger.info("Worker: container disposed")
    finally:
        try:
            await redis_client.aclose()
            logger.info("Worker: Redis disconnected")
        finally:
            await database.disconnect()
            logger.info("Worker: database disconnected")
