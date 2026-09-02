#!/usr/bin/env python3
"""Cleanup orphaned Qdrant chunks whose document_id no longer exists in PostgreSQL.

Usage:
    docker compose exec server python scripts/cleanup_orphan_chunks.py
    docker compose exec server python scripts/cleanup_orphan_chunks.py --dry-run

    ВРЕМЕННЫЙ ФАЙЛ
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Ensure the app package is importable when run from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from sqlalchemy import text


async def get_pg_document_ids(session) -> set[int]:
    """Return all document IDs from PostgreSQL."""
    result = await session.execute(text("SELECT id FROM documents"))
    return {row[0] for row in result.all()}


def get_qdrant_document_ids(client, collection: str) -> set[int]:
    """Scroll through the entire Qdrant collection and collect unique document_ids."""
    doc_ids: set[int] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=5000,
            offset=offset,
            with_payload=["metadata.document_id"],
        )
        for p in points:
            did = (p.payload or {}).get("metadata", {}).get("document_id")
            if did is not None:
                doc_ids.add(did)
        if offset is None:
            break
    return doc_ids


async def main(dry_run: bool = False) -> None:
    from infrastructure.database.database import database
    from infrastructure.qdrant_ops import create_qdrant_client
    from config import settings

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("cleanup")

    await database.connect()
    client = create_qdrant_client()
    collection = settings.collection_name

    # 1. Collect IDs from both stores
    log.info("Reading document_ids from Qdrant...")
    qdrant_ids = get_qdrant_document_ids(client, collection)
    log.info("  Qdrant: %d unique document_ids", len(qdrant_ids))

    log.info("Reading document_ids from PostgreSQL...")
    async with database.master_session() as session:
        pg_ids = await get_pg_document_ids(session)
    log.info("  PostgreSQL: %d documents", len(pg_ids))

    # 2. Diff
    orphans = qdrant_ids - pg_ids
    log.info("  Orphans: %d", len(orphans))

    if not orphans:
        log.info("Nothing to clean up.")
        return

    log.info("Orphan document_ids: %s", sorted(orphans))

    if dry_run:
        log.info("Dry-run mode — skipping deletion.")
        return

    # 3. Delete
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    for doc_id in sorted(orphans):
        client.delete(
            collection_name=collection,
            points_selector=Filter(
                must=[FieldCondition(key="metadata.document_id", match=MatchValue(value=doc_id))]
            ),
        )
        log.info("  Deleted document_id=%d", doc_id)

    log.info("Done — %d orphaned document(s) removed from Qdrant.", len(orphans))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean up orphaned Qdrant chunks")
    parser.add_argument("--dry-run", action="store_true", help="Count orphans without deleting")
    args = parser.parse_args()
    asyncio.run(main(dry_run=args.dry_run))
