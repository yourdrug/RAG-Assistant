"""Worker cron tasks — periodic maintenance jobs run by Arq worker on schedule."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from config import BenchmarkSection, JobsSection, ObservabilitySection, process_section
from infrastructure.metrics.jobs import record_job_event
from infrastructure.worker.policy import RetryPolicy, max_tries


if TYPE_CHECKING:
    from application.uow.factory import UnitOfWorkFactory
    from domain.runtime.jobs import BackgroundJob


logger = logging.getLogger("default")
CLEANUP_BATCH_SIZE = 1000


async def cron_job_cleanup(ctx: dict[str, Any]) -> None:
    """Bound each transaction and each cron run as the retained table grows."""
    uow_factory = ctx["uow_factory"]

    for _batch in range(20):
        async with uow_factory.create(master=True) as uow:
            deleted = await uow.background_jobs.delete_old(days=process_section(JobsSection).job_cleanup_days)

        if deleted:
            await record_job_event("cleaned", "all", deleted)
            logger.info("Cron: cleaned up %d old background jobs", deleted)

        if deleted < CLEANUP_BATCH_SIZE:
            break


async def cron_recover_orphaned_jobs(ctx: dict[str, Any]) -> None:
    """Release jobs whose lease expired and fail legacy pending jobs never picked up (every minute).

    A lost attempt (worker crash, OOM) is requeued with backoff while its job type's
    execution budget allows; afterwards it fails together with its document.
    """
    jobs_settings = process_section(JobsSection)

    uow_factory = ctx["uow_factory"]
    requeued: list[int] = []
    failed: list[BackgroundJob] = []
    expired_types: list[str] = []
    async with uow_factory.create(master=True) as uow:
        for job in await uow.background_jobs.lock_expired(
            timeout_minutes=jobs_settings.stuck_job_timeout_minutes
        ):
            if job.id is None:
                continue

            expired_types.append(job.job_type)

            if job.attempt < max_tries(job.job_type):
                await uow.background_jobs.release_expired(
                    job.id,
                    error="Execution lease expired; retrying",
                    retry_seconds=RetryPolicy().backoff(max(job.attempt, 1)),
                )
                requeued.append(job.id)
            else:
                await uow.background_jobs.release_expired(
                    job.id, error="Execution lease expired; attempts exhausted", retry_seconds=None
                )
                failed.append(job)

        stale_pending = await uow.background_jobs.fail_stale_pending(
            timeout_minutes=jobs_settings.stale_pending_timeout_minutes
        )
        await uow.benchmark_sweeps.fail_dead_jobs()

    if requeued:
        logger.warning("Cron: requeued %d jobs with expired leases: %s", len(requeued), requeued)

    for job_type in expired_types:
        await record_job_event("lease_expired", job_type)

    if failed:
        logger.warning(
            "Cron: failed %d jobs with expired leases: %s", len(failed), [job.id for job in failed]
        )

    for job in stale_pending:
        logger.warning("Cron: failed stale pending job %d (%s)", job.id, job.job_type)

    await fail_documents_of_dead_jobs(uow_factory, [*stale_pending, *failed])


async def fail_documents_of_dead_jobs(uow_factory: UnitOfWorkFactory, dead_jobs: list[BackgroundJob]) -> None:
    """Mark documents failed whose processing job died (never picked up or lease lost)."""
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
                    error="Processing job died — never picked up or worker lost",
                )
                logger.warning("Cron: marked document %d failed (dead job)", doc_id)


async def cron_recover_stuck_processing(ctx: dict[str, Any]) -> None:
    """Fail documents stuck in 'processing' with no active background job (every 15 min)."""
    uow_factory = ctx["uow_factory"]
    async with uow_factory.create(master=True) as uow:
        stuck_ids = await uow.documents.mark_stuck_processing_failed()

    for doc_id in stuck_ids:
        logger.warning("Cron: marked stuck PROCESSING document %d as failed", doc_id)


async def cron_chat_log_cleanup(ctx: dict[str, Any]) -> None:
    settings = process_section(ObservabilitySection)

    for _batch in range(20):
        async with ctx["uow_factory"].create(master=True) as uow:
            deleted = await uow.chat_logs.delete_expired(
                settings.chat_trace_retention_days, settings.chat_log_retention_days
            )

        if deleted < CLEANUP_BATCH_SIZE:
            break


async def cron_llm_replay_cleanup(ctx: dict[str, Any]) -> None:
    """Delete recorded benchmark LLM responses that no run reused within the retention period."""
    retention_days = process_section(BenchmarkSection).benchmark_llm_replay_retention_days
    deleted = await ctx["llm_replay_store"].purge(retention_days)

    if deleted:
        logger.info("Cron: deleted %d unused recorded LLM responses", deleted)
