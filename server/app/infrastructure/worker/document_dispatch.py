"""Deliver durable document jobs to Redis until a worker picks them up."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from infrastructure.worker.queue import QUEUE_NAME, enqueue_arq


logger = logging.getLogger(__name__)


async def cron_dispatch_documents(ctx: dict[str, Any]) -> None:
    # Keep row locks through publish: concurrent relays skip these jobs and a
    # worker cannot finish one between selection and enqueue (keep_result=0).
    # A crash before commit leaves it eligible; Arq deduplicates the same ID.
    async with ctx["uow_factory"].create(master=True) as uow:
        jobs = await uow.background_jobs.pending_dispatch()

        for job in jobs:
            if job.id is None or job.dispatch_payload is None:
                continue

            try:
                async with asyncio.timeout(5):
                    await enqueue_arq(
                        QUEUE_NAME,
                        "process_document",
                        arq_job_id=f"process_doc:{job.related_id}:{job.id}",
                        **{
                            **job.dispatch_payload,
                            "job_id": job.id,
                            "cost_operation_id": f"document_processing:{job.id}",
                        },
                    )
            except Exception:
                logger.exception("Document job %d dispatch failed; will retry", job.id)
            # Rotate attempts so a large backlog (or a failing job) cannot
            # starve later submissions. Durable jobs bypass stale-pending failure.
            await uow.background_jobs.touch_heartbeat(job.id)
