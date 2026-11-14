"""Task enqueue helpers — publish tasks to Arq (Redis-backed queue).

Redis is a mandatory component.  All background tasks are enqueued via Arq
and processed by a separate worker process.  No in-memory fallback.

Committed BackgroundJob payloads are relayed independently of Redis availability.
Redis IDs identify deliveries; active-operation deduplication and execution
ownership are enforced in PostgreSQL.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING, Any, Unpack, cast

from arq import create_pool

from application.jobs.principal import job_principal
from application.ports.job_enqueuer import DuplicateJobError
from config import RedisSection, process_section
from domain.knowledge.enums import DocumentVisibility
from domain.runtime.jobs import JobType
from infrastructure.redis.arq_settings import arq_redis_settings
from infrastructure.worker.registry import TASKS


__all__ = ["arq_redis_settings"]


if TYPE_CHECKING:
    from arq.connections import ArqRedis
    from arq.jobs import Job

    from application.ports.job_enqueuer import DocumentProcessingRequest


logger = logging.getLogger("default")


@dataclass
class PoolState:
    pool: ArqRedis | None = None


pool_state = PoolState()
pool_lock = asyncio.Lock()


async def get_pool() -> ArqRedis:
    """Lazily create and reuse one arq pool per process."""
    if pool_state.pool is None:
        async with pool_lock:
            if pool_state.pool is None:
                pool_state.pool = await create_pool(
                    arq_redis_settings(process_section(RedisSection).redis_url)
                )

    return pool_state.pool


async def close_arq_pool() -> None:
    """Close the shared arq pool (call from lifespan shutdown)."""
    async with pool_lock:
        if pool_state.pool is not None:
            pool = pool_state.pool
            pool_state.pool = None
            await pool.close()


async def enqueue_arq(
    queue_name: str, func_name: str, arq_job_id: str | None = None, **kwargs: object
) -> Job | None:
    """Enqueue a task via Arq's Redis queue.

    Returns None when arq refused the enqueue because a job with the same
    ``_job_id`` is already queued or running (duplicate suppression).
    """
    if job_principal.get() is not None and "cost_principal" not in kwargs:
        kwargs["cost_principal"] = job_principal.get()

    async with asyncio.timeout(5):
        pool = await get_pool()
        job = await pool.enqueue_job(
            func_name, _queue_name=queue_name, _job_id=arq_job_id, **cast("dict[str, Any]", kwargs)
        )

    if job is None:
        # Expected for relay re-publication: delivery IDs are per durable job.
        logger.debug(
            "Duplicate enqueue suppressed: %s (job_id=%s already queued/running)",
            func_name,
            arq_job_id,
        )
    else:
        logger.info("Enqueued task %s to queue %s (job_id=%s)", func_name, queue_name, arq_job_id)

    return job


# ---------------------------------------------------------------------------
# Public helpers — one per task type
# ---------------------------------------------------------------------------


async def enqueue_document_processing(**request: Unpack[DocumentProcessingRequest]) -> None:
    """Enqueue document processing via Arq."""
    await enqueue_arq(
        TASKS[JobType.DOCUMENT_PROCESSING].queue,
        TASKS[JobType.DOCUMENT_PROCESSING].name,
        arq_job_id=f"process_doc:{request['document_id']}:{request['job_id']}",
        **{
            **request,
            "doc_domain": request.get("doc_domain"),
            "principal_id": request.get("principal_id"),
        },
    )


async def enqueue_ingest(
    *,
    resolved_dir: str,
    reset: bool,
    domain: str,
    job_id: int,
    visibility: str = DocumentVisibility.INTERNAL_PUBLIC.value,
    group_id: int | None = None,
    client_id: int | None = None,
) -> None:
    """Enqueue full ingestion via Arq (deduplicated per parameters)."""
    job = await enqueue_arq(
        TASKS[JobType.INGEST].queue,
        TASKS[JobType.INGEST].name,
        arq_job_id=f"job:{job_id}",
        resolved_dir=resolved_dir,
        reset=reset,
        domain=domain,
        job_id=job_id,
        visibility=visibility,
        group_id=group_id,
        client_id=client_id,
    )

    if job is None:
        msg = "An equivalent ingestion is already queued or running"
        raise DuplicateJobError(msg)


async def enqueue_ingest_file(
    *,
    resolved: str,
    domain: str,
    job_id: int,
    visibility: str = DocumentVisibility.INTERNAL_PUBLIC.value,
    group_id: int | None = None,
    client_id: int | None = None,
    force: bool = False,
) -> None:
    """Enqueue single-file ingestion via Arq (deduplicated per parameters)."""
    job = await enqueue_arq(
        TASKS[JobType.INGEST_FILE].queue,
        TASKS[JobType.INGEST_FILE].name,
        arq_job_id=f"job:{job_id}",
        resolved=resolved,
        domain=domain,
        job_id=job_id,
        visibility=visibility,
        group_id=group_id,
        client_id=client_id,
        force=force,
    )

    if job is None:
        msg = "An equivalent ingestion is already queued or running"
        raise DuplicateJobError(msg)


async def enqueue_benchmark(
    *,
    dataset: str,
    top_k: int,
    judge_model: str,
    job_id: int,
) -> None:
    """Enqueue benchmark run via Arq (deduplicated per parameters)."""
    await enqueue_arq(
        TASKS[JobType.BENCHMARK].queue,
        TASKS[JobType.BENCHMARK].name,
        arq_job_id=f"job:{job_id}",
        dataset=dataset,
        top_k=top_k,
        judge_model=judge_model,
        job_id=job_id,
    )


async def enqueue_sweep(
    *,
    sweep_id: int,
    job_id: int,
) -> None:
    """Enqueue parameter sweep via Arq (deduplicated per sweep)."""
    await enqueue_arq(
        TASKS[JobType.SWEEP].queue,
        TASKS[JobType.SWEEP].name,
        arq_job_id=f"job:{job_id}",
        sweep_id=sweep_id,
        job_id=job_id,
    )


async def enqueue_document_deletion(*, document_id: int, user_id: int, job_id: int) -> None:
    job = await enqueue_arq(
        TASKS[JobType.DOCUMENT_DELETION].queue,
        TASKS[JobType.DOCUMENT_DELETION].name,
        arq_job_id=f"delete_doc:{document_id}:{job_id}",
        document_id=document_id,
        user_id=user_id,
        job_id=job_id,
    )

    if job is None:
        msg = "Document deletion job was not enqueued"
        raise RuntimeError(msg)
