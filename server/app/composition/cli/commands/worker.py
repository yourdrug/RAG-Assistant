"""CLI-команда: запуск Arq worker для обработки фоновых задач."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
import logging
import sys
from typing import Any

from arq.cron import cron
from arq.worker import Function, Worker, func as arq_func

from composition.container import Container
from composition.utils import require
from config import settings
from infrastructure.database.database import database
from infrastructure.events.postgres_config_listener import PostgresConfigListener
from infrastructure.redis.redis_client import redis_client
from infrastructure.resilience.circuit_breaker import init_breakers
from infrastructure.worker.checkpoint_cleanup import cron_sweep_checkpoint_cleanup
from infrastructure.worker.cron import (
    cron_job_cleanup,
    cron_recover_orphaned_jobs,
    cron_recover_stuck_processing,
)
from infrastructure.worker.document_deletion import delete_document
from infrastructure.worker.document_dispatch import cron_dispatch_documents
from infrastructure.worker.queue import QUEUE_NAME, arq_redis_settings, close_arq_pool
from infrastructure.worker.sweep import run_sweep_task as run_sweep
from infrastructure.worker.tasks import (
    process_document,
    run_benchmark,
    run_full_ingest,
    run_single_ingest,
)


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
    Cron-задачи: job_cleanup (2x/day), recover_orphaned (15 мин), recover_stuck_processing (15 мин).
    Использует Redis как брокер.
    """
    try:
        redis_settings = arq_redis_settings(settings.redis_url)

        functions: Sequence[Function] = [
            arq_func(delete_document, timeout=_PROCESS_TIMEOUT, keep_result=0, max_tries=_MAX_JOB_TRIES),
            arq_func(process_document, timeout=_PROCESS_TIMEOUT, keep_result=0, max_tries=_MAX_JOB_TRIES),
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
        ]

        cron_jobs = [
            cron(cron_dispatch_documents, second={0, 30}, timeout=_CRON_TIMEOUT, run_at_startup=True),
            cron(cron_sweep_checkpoint_cleanup, minute=set(range(60)), timeout=_CRON_TIMEOUT),
            cron(cron_job_cleanup, hour={1, 13}, timeout=_CRON_TIMEOUT),
            cron(cron_recover_orphaned_jobs, minute={0, 15, 30, 45}, timeout=_CRON_TIMEOUT),
            cron(cron_recover_stuck_processing, minute={5, 20, 35, 50}, timeout=_CRON_TIMEOUT),
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
            "Arq worker starting — queue: %s cron: cleanup/recover/reconcile max_jobs=%d redis=%s",
            QUEUE_NAME,
            max_jobs,
            settings.redis_host,
        )
        w.run()
    except ImportError:
        logger.exception("arq package not installed. Run: pip install arq redis")
        sys.exit(1)
    except Exception as exc:
        logger.exception("Worker startup failed", exc_info=exc)
        sys.exit(1)


async def on_startup(ctx: dict[str, Any]) -> None:
    """Initialize database and infrastructure on worker startup."""
    try:
        await database.connect()
        await redis_client.init()
        container = Container()
        ctx["container"] = container
        container.init(database)
        await container.infrastructure.bootstrap_configuration(role="worker")
        ctx.update(container.worker_dependencies())
        listener = require(container.infrastructure.events.config_listener, "worker config listener")
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


async def periodic_config_resync(listener: PostgresConfigListener) -> None:
    while True:
        await asyncio.sleep(60)
        await listener.resync(trigger="worker_periodic")


async def on_shutdown(ctx: dict[str, Any]) -> None:
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
            try:
                await close_arq_pool()
            finally:
                await redis_client.aclose()
                logger.info("Worker: Redis disconnected")
        finally:
            await database.disconnect()
            logger.info("Worker: database disconnected")
