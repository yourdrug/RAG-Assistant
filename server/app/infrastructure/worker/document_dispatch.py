"""Relay committed jobs to their registered queues; database remains the source of work."""

import logging
from typing import Any

from infrastructure.metrics.jobs import record_job_event
from infrastructure.worker.queue import enqueue_arq
from infrastructure.worker.registry import TASKS


logger = logging.getLogger(__name__)
DISPATCH_BATCH = 100
# Retries legitimately re-publish a few times; more suggests lost deliveries.
LOST_DELIVERY_WARN_AFTER = 5


def delivery_id(job_type: str, related_id: int | None, job_id: int) -> str:
    # Preserve the IDs used by in-flight jobs during rolling upgrades.
    if job_type == "document_processing":
        return f"process_doc:{related_id}:{job_id}"

    if job_type == "document_deletion":
        return f"delete_doc:{related_id}:{job_id}"

    return f"job:{job_id}"


async def cron_dispatch_documents(ctx: dict[str, Any]) -> None:
    # SKIP LOCKED serializes relays (any number of maintenance replicas) through
    # publication. A crash before commit leaves the row eligible; repeated
    # delivery is fenced again at execution. Jobs already in the broker are
    # re-published at most once per REDISPATCH_AFTER and refused as duplicates.
    async with ctx["uow_factory"].create(master=True) as uow:
        jobs = await uow.background_jobs.pending_dispatch(limit=DISPATCH_BATCH)

        for job in jobs:
            if job.id is None or job.dispatch_payload is None:
                continue

            try:
                definition = TASKS[job.job_type]
                payload = definition.validate(job.dispatch_payload, job.payload_version)
            except (KeyError, ValueError):
                await uow.background_jobs.mark_failed(
                    job.id, "Unknown task or invalid/version-unsupported payload"
                )
                logger.warning(
                    "Invalid durable job contract: job_id=%d type=%s version=%d",
                    job.id,
                    job.job_type,
                    job.payload_version,
                )
                continue

            published = False
            try:
                published = (
                    await enqueue_arq(
                        definition.queue,
                        definition.name,
                        arq_job_id=delivery_id(job.job_type, job.related_id, job.id),
                        **{**payload, "job_id": job.id},
                    )
                    is not None
                )
            except Exception as exc:  # noqa: BLE001 — committed work is retried by the relay
                logger.warning("Job dispatch deferred: job_id=%d error_type=%s", job.id, type(exc).__name__)
                await uow.background_jobs.record_dispatch(job.id, published=False)
                # The broker is unavailable: stop this round instead of waiting out
                # the publish timeout for every selected job while holding row locks.
                break

            if published and job.dispatch_attempts >= LOST_DELIVERY_WARN_AFTER:
                # The broker had no record of a job that was published before:
                # deliveries are being lost (Redis eviction/flush, expired jobs).
                logger.warning(
                    "Job re-published after lost delivery: job_id=%d publications=%d",
                    job.id,
                    job.dispatch_attempts + 1,
                )

            if published and job.dispatch_attempts > 0:
                await record_job_event("republished", job.job_type)

            await uow.background_jobs.record_dispatch(job.id, published=published)
