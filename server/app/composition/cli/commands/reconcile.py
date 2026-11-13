"""CLI command: reconcile Qdrant and Postgres document state.

Scans Qdrant for orphaned points (document_id no longer in Postgres, or a
chunk id no longer in Postgres for a live document, e.g. left by re-chunking)
and optionally deletes them.  Also detects Postgres documents missing Qdrant
chunks.  Runs as a one-shot CLI command with --dry-run (default) or
--delete to actually remove orphans.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Annotated

from sqlalchemy import text
import typer

from composition.cli.commands.ingest import run_with_infrastructure
from composition.infrastructure import InfrastructureContainer
from domain.knowledge.enums import DocumentStatus
from infrastructure.corpus.snapshot import collection_for
from infrastructure.database.database import database
from infrastructure.repositories.vector.client_types import VectorClient


logger = logging.getLogger("cli")

reconcile_app = typer.Typer(help="Reconcile Qdrant/Postgres document state")


def get_all_qdrant_points(client: VectorClient) -> dict[int, int]:
    """Scroll through the Qdrant collection: point id (= chunk id) → document_id."""
    points_by_id: dict[int, int] = {}
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=collection_for(client),
            offset=offset,
            limit=1000,
            with_payload=["metadata.document_id"],
        )

        for p in points:
            meta = p.payload or {}
            doc_id = meta.get("metadata", {}).get("document_id")

            if doc_id is not None:
                points_by_id[int(p.id)] = int(doc_id)

        if offset is None:
            break

    return points_by_id


async def get_all_postgres_doc_ids() -> tuple[set[int], set[int]]:
    """Read authoritative IDs and documents expected in Qdrant from the primary."""
    session = database.get_write_session()
    try:
        result = await session.execute(text("SELECT id, status FROM documents"))
        rows = result.fetchall()
        return {row[0] for row in rows}, {row[0] for row in rows if row[1] != DocumentStatus.FAILED.value}
    finally:
        await session.close()


async def get_all_postgres_chunk_ids() -> set[int]:
    """Read chunk ids from the primary; read after the Qdrant scan (ids are never reused)."""
    session = database.get_write_session()
    try:
        result = await session.execute(text("SELECT id FROM chunks"))
        return {row[0] for row in result.fetchall()}
    finally:
        await session.close()


def find_stale_chunk_points(
    points_by_id: dict[int, int], postgres_ids: set[int], chunk_ids: set[int]
) -> dict[int, list[int]]:
    """Points of live documents whose chunk row is gone, grouped by document_id.

    The Qdrant scan precedes the SQL read, so a point seen in Qdrant had its chunk
    committed earlier; if the chunk is missing now it was deleted, not just created.
    """
    stale: dict[int, list[int]] = {}

    for point_id, doc_id in sorted(points_by_id.items()):
        if doc_id in postgres_ids and point_id not in chunk_ids:
            stale.setdefault(doc_id, []).append(point_id)

    return stale


async def delete_stale_chunk_points(infra: InfrastructureContainer, stale: dict[int, list[int]]) -> int:
    """Delete stale chunk points under each document's index lock."""
    for doc_id, point_ids in stale.items():
        async with infra.db.uow.document_index_lock(doc_id):
            await infra.ml.vector_index_writer.delete_by_ids(point_ids)

    return sum(len(point_ids) for point_ids in stale.values())


async def delete_orphans_from_qdrant(infra: InfrastructureContainer, doc_ids: set[int]) -> int:
    """Delete all Qdrant points whose document_id is in the given set."""
    for doc_id in doc_ids:
        async with infra.db.uow.document_index_lock(doc_id):
            await infra.ml.vector_index_writer.delete_by_document_id(doc_id)

    return len(doc_ids)


@reconcile_app.command("run")
def reconcile_run(
    *,
    delete: Annotated[
        bool, typer.Option("--delete", help="Delete orphaned Qdrant points (dry-run by default)")
    ] = False,
) -> None:
    """Find and optionally clean up Qdrant/Postgres drift."""

    async def run_action(infra: InfrastructureContainer) -> int:
        # Scan and deletion must use the same bootstrapped physical generation.
        async with infra.ml.clients.corpus_operation():
            return await run_reconciliation(infra, delete=delete)

    sys.exit(asyncio.run(run_with_infrastructure(run_action)))


async def run_reconciliation(infra: InfrastructureContainer, *, delete: bool) -> int:
    client = infra.ml.clients.qdrant_client()
    logger.info("Scanning Qdrant collection '%s' ...", collection_for(client))
    points_by_id = await asyncio.to_thread(get_all_qdrant_points, client)
    qdrant_ids = set(points_by_id.values())
    logger.info("Qdrant: %d points, %d unique document_ids", len(points_by_id), len(qdrant_ids))

    logger.info("Scanning Postgres documents ...")
    postgres_ids, expected_ids = await get_all_postgres_doc_ids()
    chunk_ids = await get_all_postgres_chunk_ids()
    logger.info("Postgres: %d document_ids (%d expected in Qdrant)", len(postgres_ids), len(expected_ids))

    orphans_in_qdrant = qdrant_ids - postgres_ids
    missing_in_qdrant = expected_ids - qdrant_ids
    stale_points = find_stale_chunk_points(points_by_id, postgres_ids, chunk_ids)
    log_drift(orphans_in_qdrant, stale_points, missing_in_qdrant)

    if orphans_in_qdrant and delete:
        logger.info("Deleting %d orphaned Qdrant document groups ...", len(orphans_in_qdrant))
        deleted = await delete_orphans_from_qdrant(infra, orphans_in_qdrant)
        logger.info("Deleted orphaned points for %d documents.", deleted)

    if stale_points and delete:
        deleted = await delete_stale_chunk_points(infra, stale_points)
        logger.info("Deleted %d stale chunk points.", deleted)

    if (orphans_in_qdrant or stale_points) and not delete:
        logger.info("Dry-run: pass --delete to remove orphaned points.")

    if not orphans_in_qdrant and not missing_in_qdrant and not stale_points:
        logger.info("Reconciliation: OK — Qdrant and Postgres are in sync.")
        return 0

    return 1


def log_drift(
    orphans_in_qdrant: set[int], stale_points: dict[int, list[int]], missing_in_qdrant: set[int]
) -> None:
    if orphans_in_qdrant:
        logger.warning(
            "Orphans in Qdrant (no Postgres doc): %d — doc_ids: %s",
            len(orphans_in_qdrant),
            sorted(orphans_in_qdrant),
        )
    else:
        logger.info("No orphaned Qdrant points found.")

    if stale_points:
        logger.warning(
            "Stale chunk points in Qdrant (chunk deleted from Postgres): %d in %d documents — doc_ids: %s",
            sum(len(point_ids) for point_ids in stale_points.values()),
            len(stale_points),
            sorted(stale_points),
        )
    else:
        logger.info("No stale chunk points found.")

    if missing_in_qdrant:
        logger.warning(
            "Postgres docs missing from Qdrant: %d — doc_ids: %s",
            len(missing_in_qdrant),
            sorted(missing_in_qdrant),
        )
    else:
        logger.info("All Postgres docs have Qdrant points.")
