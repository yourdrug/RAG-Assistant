"""Task enqueue helpers — publish tasks to Arq (Redis-backed queue).

Redis is a mandatory component.  All background tasks are enqueued via Arq
and processed by a separate worker process.  No in-memory fallback.

H-7/H-8 hardening:
- A single shared arq pool is reused across enqueues (no per-call pool churn).
- Deterministic ``_job_id`` values make arq deduplicate concurrent duplicate
  enqueues (double-click / retry): the second enqueue is refused while the
  first job is still queued or running.  Jobs that were refused never run —
  the stale-pending sweeper (cron_recover_orphaned_jobs) fails such rows so
  they are visible instead of staying 'pending' forever.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
import hashlib
import logging
from typing import TYPE_CHECKING, Any, Unpack, cast
from urllib.parse import unquote
from uuid import uuid4

from arq import create_pool
from arq.connections import RedisSettings

from application.budget import billing_principal
from application.ports.job_enqueuer import DuplicateJobError
from config import RedisSection, process_section
from domain.knowledge.enums import DocumentVisibility


if TYPE_CHECKING:
    from arq.connections import ArqRedis
    from arq.jobs import Job

    from application.ports.job_enqueuer import DocumentProcessingRequest


logger = logging.getLogger("default")

QUEUE_NAME = "background_tasks"


@dataclass
class PoolState:
    pool: ArqRedis | None = None


pool_state = PoolState()
pool_lock = asyncio.Lock()


def arq_redis_settings(dsn: str) -> RedisSettings:
    """Build arq settings from a DSN, decoding percent-encoded credentials like redis-py does."""
    parsed = RedisSettings.from_dsn(dsn)
    return replace(
        parsed,
        username=unquote(parsed.username) if parsed.username else parsed.username,
        password=unquote(parsed.password) if parsed.password else parsed.password,
    )


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


def deterministic_id(prefix: str, *parts: object) -> str:
    raw = "|".join(str(p) for p in parts)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]  # noqa: S324
    return f"{prefix}:{digest}"


async def enqueue_arq(
    queue_name: str, func_name: str, arq_job_id: str | None = None, **kwargs: object
) -> Job | None:
    """Enqueue a task via Arq's Redis queue.

    Returns None when arq refused the enqueue because a job with the same
    ``_job_id`` is already queued or running (duplicate suppression).
    """
    if billing_principal.get() is not None:
        kwargs["cost_principal"] = billing_principal.get()

    kwargs.setdefault("cost_operation_id", str(uuid4()))
    pool = await get_pool()
    job = await pool.enqueue_job(
        func_name, _queue_name=queue_name, _job_id=arq_job_id, **cast("dict[str, Any]", kwargs)
    )

    if job is None:
        logger.warning(
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
        QUEUE_NAME,
        "process_document",
        arq_job_id=f"process_doc:{request['document_id']}:{request['job_id']}",
        **{**request, "doc_domain": request.get("doc_domain"), "principal_id": request.get("principal_id")},
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
        QUEUE_NAME,
        "run_full_ingest",
        arq_job_id=deterministic_id(
            "ingest_full", resolved_dir, reset, domain, visibility, group_id, client_id
        ),
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
        QUEUE_NAME,
        "run_single_ingest",
        arq_job_id=deterministic_id("ingest_file", resolved, domain, visibility, group_id, client_id, force),
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
        QUEUE_NAME,
        "run_benchmark",
        arq_job_id=deterministic_id("benchmark", job_id),
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
        QUEUE_NAME,
        "run_sweep",
        arq_job_id=f"sweep:{sweep_id}",
        sweep_id=sweep_id,
        job_id=job_id,
    )


async def enqueue_document_deletion(*, document_id: int, user_id: int, job_id: int) -> None:
    job = await enqueue_arq(
        QUEUE_NAME,
        "delete_document",
        arq_job_id=f"delete_doc:{document_id}:{job_id}",
        document_id=document_id,
        user_id=user_id,
        job_id=job_id,
    )

    if job is None:
        msg = "Document deletion job was not enqueued"
        raise RuntimeError(msg)
