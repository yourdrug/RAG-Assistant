"""Worker cron tasks — periodic maintenance jobs run by Arq worker on schedule."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from config import JobsSection, process_section


if TYPE_CHECKING:
    from application.uow.factory import UnitOfWorkFactory
    from domain.runtime.jobs import BackgroundJob


logger = logging.getLogger("default")


async def cron_job_cleanup(ctx: dict[str, Any]) -> None:
    """Delete old background job records (runs every hour)."""
    uow_factory = ctx["uow_factory"]
    async with uow_factory.create(master=True) as uow:
        deleted = await uow.background_jobs.delete_old(days=process_section(JobsSection).job_cleanup_days)

        if deleted:
            logger.info("Cron: cleaned up %d old background jobs", deleted)


async def cron_recover_orphaned_jobs(ctx: dict[str, Any]) -> None:
    """Fail jobs whose heartbeat expired and pending jobs never picked up (every 15 min)."""
    jobs_settings = process_section(JobsSection)

    uow_factory = ctx["uow_factory"]
    async with uow_factory.create(master=True) as uow:
        orphaned_ids = await uow.background_jobs.recover_orphaned(
            timeout_minutes=jobs_settings.stuck_job_timeout_minutes
        )
        stale_pending = await uow.background_jobs.fail_stale_pending(
            timeout_minutes=jobs_settings.stale_pending_timeout_minutes
        )
        await uow.benchmark_sweeps.fail_dead_jobs()

    if orphaned_ids:
        logger.warning("Cron: recovered %d orphaned jobs: %s", len(orphaned_ids), orphaned_ids)

    for job in stale_pending:
        logger.warning("Cron: failed stale pending job %d (%s)", job.id, job.job_type)

    await fail_documents_of_dead_jobs(uow_factory, stale_pending)


async def fail_documents_of_dead_jobs(uow_factory: UnitOfWorkFactory, dead_jobs: list[BackgroundJob]) -> None:
    """Mark documents failed whose processing job died before pickup."""
    doc_ids = [
        job.related_id for job in dead_jobs if job.job_type == "document_processing" and job.related_id
    ]

    if not doc_ids:
        return

    async with uow_factory.create(master=True) as uow:
        for doc_id in doc_ids:
            doc = await uow.documents.get_by_id(doc_id)

            if doc is not None and doc.status.value in ("pending", "processing"):
                await uow.documents.update_status(
                    doc_id,
                    status="failed",
                    error="Processing job was never picked up — enqueue lost or worker unavailable",
                )
                logger.warning("Cron: marked document %d failed (dead job)", doc_id)


async def cron_recover_stuck_processing(ctx: dict[str, Any]) -> None:
    """Fail documents stuck in 'processing' with no active background job (every 15 min)."""
    uow_factory = ctx["uow_factory"]
    async with uow_factory.create(master=True) as uow:
        stuck_ids = await uow.documents.mark_stuck_processing_failed()

    for doc_id in stuck_ids:
        logger.warning("Cron: marked stuck PROCESSING document %d as failed", doc_id)
