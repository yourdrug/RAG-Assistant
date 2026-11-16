"""Infrastructure adapter for JobEnqueuerPort — wraps worker.queue functions."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Unpack

from arq.jobs import Job
from redis.exceptions import RedisError

from domain.knowledge.enums import DocumentVisibility
from domain.runtime.jobs import JobType
from infrastructure.worker.document_dispatch import delivery_id
from infrastructure.worker.queue import (
    enqueue_benchmark as _enqueue_benchmark,
    enqueue_document_deletion,
    enqueue_document_processing as _enqueue_document_processing,
    enqueue_ingest as _enqueue_ingest,
    enqueue_ingest_file as _enqueue_ingest_file,
    enqueue_sweep as _enqueue_sweep,
    get_pool,
)
from infrastructure.worker.registry import TASKS


if TYPE_CHECKING:
    from application.ports.job_enqueuer import DocumentProcessingRequest


class JobEnqueuerAdapter:
    """Thin wrapper making worker queue functions available as an injectable port."""

    async def abort_job(self, job_id: int, job_type: str, related_id: int | None) -> None:
        try:
            async with asyncio.timeout(5):
                pool = await get_pool()
                job = Job(
                    delivery_id(job_type, related_id, job_id),
                    pool,
                    _queue_name=TASKS[JobType(job_type)].queue,
                )
                try:
                    await job.abort(timeout=1, poll_delay=0.1)
                except TimeoutError:
                    # keep_result=0: the request is written without a retained result.
                    return
        except RedisError as exc:
            message = "Broker abort unavailable"
            raise ConnectionError(message) from exc

    async def enqueue_document_deletion(self, *, document_id: int, user_id: int, job_id: int) -> None:
        await enqueue_document_deletion(document_id=document_id, user_id=user_id, job_id=job_id)

    async def enqueue_document_processing(self, **request: Unpack[DocumentProcessingRequest]) -> None:
        await _enqueue_document_processing(**request)

    async def enqueue_ingest(  # noqa: PLR0913 - keyword-only ingestion scope of one job
        self,
        *,
        resolved_dir: str,
        reset: bool,
        domain: str,
        job_id: int,
        visibility: str = DocumentVisibility.INTERNAL_PUBLIC.value,
        group_id: int | None = None,
        client_id: int | None = None,
        space_id: int | None = None,
    ) -> None:
        await _enqueue_ingest(
            resolved_dir=resolved_dir,
            reset=reset,
            domain=domain,
            job_id=job_id,
            visibility=visibility,
            group_id=group_id,
            client_id=client_id,
            space_id=space_id,
        )

    async def enqueue_ingest_file(  # noqa: PLR0913 - keyword-only ingestion scope of one job
        self,
        *,
        resolved: str,
        domain: str,
        job_id: int,
        visibility: str = DocumentVisibility.INTERNAL_PUBLIC.value,
        group_id: int | None = None,
        client_id: int | None = None,
        space_id: int | None = None,
        force: bool = False,
    ) -> None:
        await _enqueue_ingest_file(
            resolved=resolved,
            domain=domain,
            job_id=job_id,
            visibility=visibility,
            group_id=group_id,
            client_id=client_id,
            space_id=space_id,
            force=force,
        )

    async def enqueue_benchmark(
        self,
        *,
        dataset: str,
        top_k: int,
        judge_model: str,
        job_id: int,
        space_id: int | None = None,
    ) -> None:
        await _enqueue_benchmark(
            dataset=dataset,
            top_k=top_k,
            judge_model=judge_model,
            job_id=job_id,
            space_id=space_id,
        )

    async def enqueue_sweep(
        self,
        *,
        sweep_id: int,
        job_id: int,
    ) -> None:
        await _enqueue_sweep(sweep_id=sweep_id, job_id=job_id)
