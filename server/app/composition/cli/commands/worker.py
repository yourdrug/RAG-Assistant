"""CLI-команда: запуск Arq worker для обработки фоновых задач."""

from __future__ import annotations

import asyncio
from enum import StrEnum
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
    cron_chat_log_cleanup,
    cron_job_cleanup,
    cron_llm_replay_cleanup,
    cron_recover_orphaned_jobs,
    cron_recover_stuck_processing,
)
from infrastructure.worker.document_dispatch import cron_dispatch_documents
from infrastructure.worker.maintenance import cron_outbox_dispatch, cron_outbox_reconcile
from infrastructure.worker.queue import arq_redis_settings, close_arq_pool
from infrastructure.worker.registry import BATCH_QUEUE, DOCUMENTS_QUEUE, MAINTENANCE_QUEUE, tasks_for_queue
from infrastructure.worker_health import health_check_key


logger = logging.getLogger("cli")

CRON_TIMEOUT = 60 * 10


class WorkerRole(StrEnum):
    DOCUMENTS = "documents"
    BATCH = "batch"
    MAINTENANCE = "maintenance"


QUEUES = {
    WorkerRole.DOCUMENTS: DOCUMENTS_QUEUE,
    WorkerRole.BATCH: BATCH_QUEUE,
    WorkerRole.MAINTENANCE: MAINTENANCE_QUEUE,
}
# Busy concurrency groups defer deliveries with Arq Retry; DB execution attempts,
# not broker deliveries, enforce the retry budget, so this is only a safety net.
MAX_DELIVERIES = 1000


def default_max_jobs(role: WorkerRole) -> int:
    """Broker slots per replica; total capacity scales with the replica count."""
    if role == WorkerRole.MAINTENANCE:
        return settings.maintenance_max_concurrent

    if role == WorkerRole.DOCUMENTS:
        return settings.document_max_concurrent

    # Batch slots per replica; cluster-wide caps per job type apply at claim.
    return settings.worker_max_concurrent


def worker(
    max_jobs: int | None = None,
    health_check_interval: int = 10,
    role: WorkerRole = WorkerRole.DOCUMENTS,
) -> None:
    """Run one Arq queue: documents, batch or maintenance (cron, relay, outbox)."""
    role = WorkerRole(role)
    try:
        redis_settings = arq_redis_settings(settings.redis_url)

        maintenance = role == WorkerRole.MAINTENANCE
        queue_name = QUEUES[role]
        functions: list[Function] = [
            arq_func(
                task.handler, name=task.name, timeout=task.timeout, keep_result=0, max_tries=MAX_DELIVERIES
            )
            for task in tasks_for_queue(queue_name)
        ]
        cron_jobs = (
            [
                cron(
                    cron_dispatch_documents,
                    job_id="maintenance:cron_dispatch_documents",
                    second={0, 30},
                    timeout=CRON_TIMEOUT,
                    run_at_startup=True,
                ),
                cron(
                    cron_outbox_dispatch,
                    job_id="maintenance:cron_outbox_dispatch",
                    second={5, 35},
                    timeout=CRON_TIMEOUT,
                    run_at_startup=True,
                ),
                cron(
                    cron_outbox_reconcile,
                    job_id="maintenance:cron_outbox_reconcile",
                    minute=set(range(60)),
                    second=10,
                    timeout=CRON_TIMEOUT,
                ),
                cron(
                    cron_sweep_checkpoint_cleanup,
                    job_id="maintenance:cron_sweep_checkpoint_cleanup",
                    minute=set(range(60)),
                    timeout=CRON_TIMEOUT,
                ),
                cron(
                    cron_chat_log_cleanup,
                    job_id="maintenance:cron_chat_log_cleanup",
                    minute=15,
                    timeout=CRON_TIMEOUT,
                ),
                cron(
                    cron_llm_replay_cleanup,
                    job_id="maintenance:cron_llm_replay_cleanup",
                    minute=45,
                    hour=3,
                    timeout=CRON_TIMEOUT,
                ),
                cron(
                    cron_job_cleanup,
                    job_id="maintenance:cron_job_cleanup",
                    minute=0,
                    timeout=CRON_TIMEOUT,
                ),
                cron(
                    cron_recover_orphaned_jobs,
                    job_id="maintenance:cron_recover_orphaned_jobs",
                    # Leases last minutes; recover lost attempts promptly.
                    minute=set(range(60)),
                    second=20,
                    timeout=CRON_TIMEOUT,
                ),
                cron(
                    cron_recover_stuck_processing,
                    job_id="maintenance:cron_recover_stuck_processing",
                    minute={5, 20, 35, 50},
                    timeout=CRON_TIMEOUT,
                ),
            ]
            if maintenance
            else []
        )

        if max_jobs is None:
            max_jobs = default_max_jobs(role)

        w = Worker(
            functions=functions,
            cron_jobs=cron_jobs,
            redis_settings=redis_settings,
            max_jobs=max_jobs,
            health_check_interval=health_check_interval,
            queue_name=queue_name,
            health_check_key=health_check_key(queue_name),
            job_completion_wait=settings.worker_completion_wait_seconds,
            allow_abort_jobs=True,
            # Results are never read; a kept failure result (e.g. a delivery to a
            # replica of the previous release) would block re-publication by ID.
            keep_result=0,
            on_startup=on_maintenance_startup if maintenance else on_startup,
            on_shutdown=on_shutdown,
        )

        logger.info(
            "Arq worker starting — role: %s queue: %s max_jobs=%d redis=%s",
            role,
            queue_name,
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

        if ctx.get("maintenance"):
            ctx["uow_factory"] = container.infrastructure.db.uow
            ctx["outbox_dispatcher"] = container.infrastructure.events.outbox_dispatcher
            # A killed predecessor in this container cannot finish its claims.
            await ctx["outbox_dispatcher"].recover_abandoned()
            outbox_listener = require(
                container.infrastructure.events.outbox_listener, "maintenance outbox listener"
            )
            await outbox_listener.start()
        else:
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


async def on_maintenance_startup(ctx: dict[str, Any]) -> None:
    ctx["maintenance"] = True
    await on_startup(ctx)


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
