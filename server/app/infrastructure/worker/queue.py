"""Task enqueue helpers — publish tasks to Arq (Redis-backed queue).

Redis is a mandatory component.  All background tasks are enqueued via Arq
and processed by a separate worker process.  No in-memory fallback.
"""

from __future__ import annotations

import logging
from typing import Any

from arq import create_pool
from arq.connections import RedisSettings
from config import settings

logger = logging.getLogger("default")


async def _enqueue_arq(queue_name: str, func_name: str, **kwargs: Any) -> None:
    """Enqueue a task via Arq's Redis queue."""
    redis_settings = RedisSettings.from_dsn(settings.redis_url)

    pool = await create_pool(redis_settings)
    try:
        await pool.enqueue_job(func_name, _queue_name=queue_name, **kwargs)
        logger.info("Enqueued task %s to queue %s", func_name, queue_name)
    finally:
        await pool.close()


# ---------------------------------------------------------------------------
# Public helpers — one per task type
# ---------------------------------------------------------------------------

_QUEUE_NAME = "document_processing"


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
) -> None:
    """Enqueue document processing via Arq."""
    await _enqueue_arq(
        _QUEUE_NAME,
        "process_document",
        document_id=document_id,
        storage_key=storage_key,
        filename=filename,
        visibility=visibility,
        owner_id=owner_id,
        group_id=group_id,
        replace_id=replace_id,
        job_id=job_id,
        doc_domain=doc_domain,
    )


async def enqueue_ingest(
    *,
    resolved_dir: str,
    reset: bool,
    domain: str,
    job_id: int,
) -> None:
    """Enqueue full ingestion via Arq."""
    await _enqueue_arq(
        _QUEUE_NAME,
        "run_full_ingest",
        resolved_dir=resolved_dir,
        reset=reset,
        domain=domain,
        job_id=job_id,
    )


async def enqueue_ingest_file(
    *,
    resolved: str,
    domain: str,
    job_id: int,
) -> None:
    """Enqueue single-file ingestion via Arq."""
    await _enqueue_arq(
        _QUEUE_NAME,
        "run_single_ingest",
        resolved=resolved,
        domain=domain,
        job_id=job_id,
    )


async def enqueue_benchmark(
    *,
    questions_path: str,
    out_dir: str,
    top_k: int,
    judge_model: str,
    job_id: int,
) -> None:
    """Enqueue benchmark run via Arq."""
    await _enqueue_arq(
        _QUEUE_NAME,
        "run_benchmark",
        questions_path=questions_path,
        out_dir=out_dir,
        top_k=top_k,
        judge_model=judge_model,
        job_id=job_id,
    )
