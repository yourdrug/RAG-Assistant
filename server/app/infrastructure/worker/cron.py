"""Worker cron tasks — periodic maintenance jobs run by Arq worker on schedule."""

from __future__ import annotations

import logging
import time
from typing import Any

from config import settings

logger = logging.getLogger("default")


async def cron_job_cleanup(ctx: dict[str, Any]) -> None:
    """Delete old background job records (runs every hour)."""
    uow_factory = ctx["container"].infrastructure.db.uow_factory
    async with uow_factory.create(master=True) as uow:
        deleted = await uow.background_jobs.delete_old(days=settings.job_cleanup_days)
        if deleted:
            logger.info("Cron: cleaned up %d old background jobs", deleted)


async def cron_recover_orphaned_jobs(ctx: dict[str, Any]) -> None:
    """Fail jobs whose heartbeat expired and pending jobs never picked up (every 15 min)."""
    uow_factory = ctx["container"].infrastructure.db.uow_factory
    async with uow_factory.create(master=True) as uow:
        orphaned_ids = await uow.background_jobs.recover_orphaned(
            timeout_minutes=settings.stuck_job_timeout_minutes
        )
        stale_pending = await uow.background_jobs.fail_stale_pending(
            timeout_minutes=settings.stale_pending_timeout_minutes
        )
    if orphaned_ids:
        logger.warning("Cron: recovered %d orphaned jobs: %s", len(orphaned_ids), orphaned_ids)
    for job in stale_pending:
        logger.warning("Cron: failed stale pending job %d (%s)", job.id, job.job_type)
    await _fail_documents_of_dead_jobs(uow_factory, stale_pending)


async def _fail_documents_of_dead_jobs(uow_factory, dead_jobs) -> None:
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
                    "failed",
                    error="Processing job was never picked up — enqueue lost or worker unavailable",
                )
                logger.warning("Cron: marked document %d failed (dead job)", doc_id)


async def cron_recover_stuck_processing(ctx: dict[str, Any]) -> None:
    """Fail documents stuck in 'processing' with no active background job (every 15 min)."""
    uow_factory = ctx["container"].infrastructure.db.uow_factory
    async with uow_factory.create(master=True) as uow:
        stuck_ids = await uow.documents.mark_stuck_processing_failed()
    for doc_id in stuck_ids:
        logger.warning("Cron: marked stuck PROCESSING document %d as failed", doc_id)


async def cron_bm25_rebuild(ctx: dict[str, Any]) -> None:
    """Rebuild BM25 index from scratch (runs daily at 3:00 AM UTC).

    Loads chunks in batches of 5000 to avoid 100MB+ memory spikes on large corpora.
    Includes ACL metadata (visibility, owner_id, group_id) for pre-filtering.
    """
    if not settings.hybrid_enabled:
        logger.debug("Cron: BM25 rebuild skipped — hybrid search disabled")
        return

    from infrastructure.bm25.bm25_invalidation import publish_bm25_invalidation
    from infrastructure.bm25.bm25_index import BM25Index
    from infrastructure.bm25.persistence import save_bm25_index_to_s3
    from infrastructure.metrics.metrics import BM25_REBUILD_MEMORY
    from infrastructure.storage import get_storage

    uow_factory = ctx["container"].infrastructure.db.uow_factory
    t0 = time.monotonic()

    all_texts: list[str] = []
    all_visibilities: list[str | None] = []
    all_owner_ids: list[int | None] = []
    all_group_ids: list[int | None] = []

    async with uow_factory.create(master=True) as uow:
        batches = await uow.chunks.get_all_contents_batches_with_acl(batch_size=5000)
        for batch in batches:
            for content, visibility, owner_id, group_id in batch:
                all_texts.append(content)
                all_visibilities.append(visibility)
                all_owner_ids.append(owner_id)
                all_group_ids.append(group_id)

    if not all_texts:
        logger.info("Cron: BM25 rebuild — no chunks found, skipping")
        return

    bm25_index = BM25Index(
        all_texts,
        doc_visibility=all_visibilities,
        doc_owner_id=all_owner_ids,
        doc_group_id=all_group_ids,
    )

    import sys

    BM25_REBUILD_MEMORY.set(sys.getsizeof(all_texts) + sys.getsizeof(bm25_index.hashes))

    storage = get_storage()
    await save_bm25_index_to_s3(bm25_index, storage)
    await publish_bm25_invalidation()

    elapsed = time.monotonic() - t0
    logger.info(
        "Cron: BM25 rebuild completed — %d chunks indexed in %.1fs (batched loading, ACL metadata)",
        len(all_texts),
        elapsed,
    )
