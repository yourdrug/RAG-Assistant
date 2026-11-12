"""CLI command: reconcile Qdrant and Postgres document state.

Scans Qdrant for orphaned points (document_id no longer in Postgres) and
optionally deletes them.  Also detects Postgres documents missing Qdrant
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


def get_all_qdrant_doc_ids(client: VectorClient) -> set[int]:
    """Scroll through Qdrant collection and extract unique document_ids."""
    doc_ids: set[int] = set()
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
                doc_ids.add(int(doc_id))

        if offset is None:
            break

    return doc_ids


async def get_all_postgres_doc_ids() -> tuple[set[int], set[int]]:
    """Read authoritative IDs and documents expected in Qdrant from the primary."""
    session = database.get_write_session()
    try:
        result = await session.execute(text("SELECT id, status FROM documents"))
        rows = result.fetchall()
        return {row[0] for row in rows}, {row[0] for row in rows if row[1] != DocumentStatus.FAILED.value}
    finally:
        await session.close()


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
    qdrant_ids = await asyncio.to_thread(get_all_qdrant_doc_ids, client)
    logger.info("Qdrant: %d unique document_ids", len(qdrant_ids))

    logger.info("Scanning Postgres documents ...")
    postgres_ids, expected_ids = await get_all_postgres_doc_ids()
    logger.info("Postgres: %d document_ids (%d expected in Qdrant)", len(postgres_ids), len(expected_ids))

    orphans_in_qdrant = qdrant_ids - postgres_ids
    missing_in_qdrant = expected_ids - qdrant_ids

    if orphans_in_qdrant:
        logger.warning(
            "Orphans in Qdrant (no Postgres doc): %d — doc_ids: %s",
            len(orphans_in_qdrant),
            sorted(orphans_in_qdrant),
        )
    else:
        logger.info("No orphaned Qdrant points found.")

    if missing_in_qdrant:
        logger.warning(
            "Postgres docs missing from Qdrant: %d — doc_ids: %s",
            len(missing_in_qdrant),
            sorted(missing_in_qdrant),
        )
    else:
        logger.info("All Postgres docs have Qdrant points.")

    if orphans_in_qdrant and delete:
        logger.info("Deleting %d orphaned Qdrant document groups ...", len(orphans_in_qdrant))
        deleted = await delete_orphans_from_qdrant(infra, orphans_in_qdrant)
        logger.info("Deleted orphaned points for %d documents.", deleted)
    elif orphans_in_qdrant:
        logger.info("Dry-run: pass --delete to remove orphaned points.")

    if not orphans_in_qdrant and not missing_in_qdrant:
        logger.info("Reconciliation: OK — Qdrant and Postgres are in sync.")
        return 0

    return 1
