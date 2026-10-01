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

import hashlib
import logging
from typing import Any

from arq import create_pool
from arq.connections import RedisSettings
from arq.jobs import Job
from config import settings

logger = logging.getLogger("default")

QUEUE_NAME = "background_tasks"

_pool = None


async def _get_pool():
    """Lazily create and reuse one arq pool per process."""
    global _pool
    if _pool is None:
        _pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    return _pool


async def close_arq_pool() -> None:
    """Close the shared arq pool (call from lifespan shutdown)."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def _deterministic_id(prefix: str, *parts: Any) -> str:
    raw = "|".join(str(p) for p in parts)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]  # noqa: S324
    return f"{prefix}:{digest}"


async def _enqueue_arq(
    queue_name: str, func_name: str, arq_job_id: str | None = None, **kwargs: Any
) -> Job | None:
    """Enqueue a task via Arq's Redis queue.

    Returns None when arq refused the enqueue because a job with the same
    ``_job_id`` is already queued or running (duplicate suppression).
    """
    pool = await _get_pool()
    job = await pool.enqueue_job(func_name, _queue_name=queue_name, _job_id=arq_job_id, **kwargs)
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


async def enqueue_document_processing(
    *,
    document_id: int,
    storage_key: str,
    filename: str,
    visibility: str,
    owner_id: int | None,
    group_id: int | None,
    replace_id: int | None,
    job_id: int,
    doc_domain: str | None = None,
    principal_id: int | None = None,
) -> None:
    """Enqueue document processing via Arq."""
    await _enqueue_arq(
        QUEUE_NAME,
        "process_document",
        arq_job_id=f"process_doc:{document_id}:{job_id}",
        document_id=document_id,
        storage_key=storage_key,
        filename=filename,
        visibility=visibility,
        owner_id=owner_id,
        group_id=group_id,
        replace_id=replace_id,
        job_id=job_id,
        doc_domain=doc_domain,
        principal_id=principal_id,
    )


async def enqueue_ingest(
    *,
    resolved_dir: str,
    reset: bool,
    domain: str,
    job_id: int,
    visibility: str = "internal_public",
    group_id: int | None = None,
    client_id: int | None = None,
) -> None:
    """Enqueue full ingestion via Arq (deduplicated per parameters)."""
    await _enqueue_arq(
        QUEUE_NAME,
        "run_full_ingest",
        arq_job_id=_deterministic_id("ingest_full", resolved_dir, reset, domain),
        resolved_dir=resolved_dir,
        reset=reset,
        domain=domain,
        job_id=job_id,
        visibility=visibility,
        group_id=group_id,
        client_id=client_id,
    )


async def enqueue_ingest_file(
    *,
    resolved: str,
    domain: str,
    job_id: int,
    visibility: str = "internal_public",
    group_id: int | None = None,
    client_id: int | None = None,
) -> None:
    """Enqueue single-file ingestion via Arq (deduplicated per parameters)."""
    await _enqueue_arq(
        QUEUE_NAME,
        "run_single_ingest",
        arq_job_id=_deterministic_id("ingest_file", resolved, domain),
        resolved=resolved,
        domain=domain,
        job_id=job_id,
        visibility=visibility,
        group_id=group_id,
        client_id=client_id,
    )


async def enqueue_benchmark(
    *,
    questions_path: str,
    out_dir: str,
    top_k: int,
    judge_model: str,
    job_id: int,
) -> None:
    """Enqueue benchmark run via Arq (deduplicated per parameters)."""
    await _enqueue_arq(
        QUEUE_NAME,
        "run_benchmark",
        arq_job_id=_deterministic_id("benchmark", questions_path, out_dir, top_k, judge_model),
        questions_path=questions_path,
        out_dir=out_dir,
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
    await _enqueue_arq(
        QUEUE_NAME,
        "run_sweep",
        arq_job_id=f"sweep:{sweep_id}",
        sweep_id=sweep_id,
        job_id=job_id,
    )
