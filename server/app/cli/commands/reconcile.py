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

import typer
from config import settings
from infrastructure.clients import get_qdrant_client
from infrastructure.database.database import database
from qdrant_client.models import FieldCondition, Filter, MatchValue

logger = logging.getLogger("cli")

reconcile_app = typer.Typer(help="Reconcile Qdrant/Postgres document state")


def _get_all_qdrant_doc_ids() -> set[int]:
    """Scroll through Qdrant collection and extract unique document_ids."""
    client = get_qdrant_client()
    doc_ids: set[int] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=settings.collection_name,
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


async def _get_all_postgres_doc_ids() -> set[int]:
    """Query Postgres for all document IDs with status != 'failed'."""
    await database.connect()
    try:
        rows = await database.fetch_all("SELECT id FROM documents WHERE status != 'failed'")
        return {row["id"] for row in rows}
    finally:
        await database.disconnect()


def _delete_orphans_from_qdrant(doc_ids: set[int]) -> int:
    """Delete all Qdrant points whose document_id is in the given set."""
    client = get_qdrant_client()
    total_deleted = 0
    for doc_id in doc_ids:
        client.delete(
            collection_name=settings.collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="metadata.document_id", match=MatchValue(value=doc_id))]
            ),
        )
        total_deleted += 1
    return total_deleted


@reconcile_app.command("run")
def reconcile_run(
    delete: bool = typer.Option(False, "--delete", help="Delete orphaned Qdrant points (dry-run by default)"),
) -> None:
    """Find and optionally clean up Qdrant/Postgres drift."""
    logger.info("Scanning Qdrant collection '%s' ...", settings.collection_name)
    qdrant_ids = _get_all_qdrant_doc_ids()
    logger.info("Qdrant: %d unique document_ids", len(qdrant_ids))

    logger.info("Scanning Postgres documents ...")
    postgres_ids = asyncio.run(_get_all_postgres_doc_ids())
    logger.info("Postgres: %d active document_ids", len(postgres_ids))

    orphans_in_qdrant = qdrant_ids - postgres_ids
    missing_in_qdrant = postgres_ids - qdrant_ids

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
        deleted = _delete_orphans_from_qdrant(orphans_in_qdrant)
        logger.info("Deleted orphaned points for %d documents.", deleted)
    elif orphans_in_qdrant:
        logger.info("Dry-run: pass --delete to remove orphaned points.")

    if not orphans_in_qdrant and not missing_in_qdrant:
        logger.info("Reconciliation: OK — Qdrant and Postgres are in sync.")
        sys.exit(0)
    else:
        sys.exit(1)
