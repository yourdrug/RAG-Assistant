"""CLI command: backfill chunk IDs to sync Postgres and Qdrant.

This command re-indexes all documents so that Qdrant point IDs match
Postgres chunk IDs. This is required for the chunk editing feature
to work correctly.

Usage:
    python main.py backfill-chunk-ids run [--dry-run] [--document-id ID]
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time

import typer
from config import settings
from infrastructure.database.database import database
from infrastructure.ml.factories import create_embeddings, create_qdrant_client
from infrastructure.ml.hybrid import content_hash
from qdrant_client.models import PointStruct

logger = logging.getLogger("cli")

backfill_app = typer.Typer(help="Backfill chunk IDs to sync Postgres and Qdrant")


async def _get_documents_with_chunks(document_id: int | None = None) -> list[dict]:
    """Get all documents with their chunks from Postgres."""
    from sqlalchemy import select

    from infrastructure.database.models import ChunkModel, DocumentModel

    await database.connect()
    try:
        async with database.get_read_session() as session:
            stmt = select(DocumentModel).where(DocumentModel.status == "done")
            if document_id:
                stmt = stmt.where(DocumentModel.id == document_id)
            doc_rows = (await session.execute(stmt)).scalars().all()

            result: list[dict] = []
            for d in doc_rows:
                cstmt = (
                    select(
                        ChunkModel.id,
                        ChunkModel.content,
                        ChunkModel.chunk_index,
                        ChunkModel.edited_at,
                    )
                    .where(ChunkModel.document_id == d.id)
                    .order_by(ChunkModel.chunk_index)
                )
                chunk_rows = (await session.execute(cstmt)).all()
                result.append(
                    {
                        "document_id": d.id,
                        "filename": d.filename,
                        "visibility": d.visibility,
                        "owner_id": d.owner_id,
                        "group_id": d.group_id,
                        "doc_domain": d.doc_domain,
                        "chunks": [
                            {
                                "id": c.id,
                                "content": c.content,
                                "chunk_index": c.chunk_index,
                                "edited_at": c.edited_at,
                            }
                            for c in chunk_rows
                        ],
                    }
                )
            return result
    finally:
        await database.disconnect()


def _delete_document_points(document_id: int) -> None:
    """Delete all Qdrant points for a document."""
    client = create_qdrant_client()
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    client.delete(
        collection_name=settings.collection_name,
        points_selector=Filter(
            must=[FieldCondition(key="metadata.document_id", match=MatchValue(value=document_id))]
        ),
    )


def _upsert_chunk(
    chunk_id: int,
    content: str,
    doc: dict,
    edited_at=None,
) -> None:
    """Upsert a single chunk to Qdrant with deterministic ID."""
    client = create_qdrant_client()
    embeddings = create_embeddings()

    # Generate embedding
    vector = embeddings.embed_query_sync(content)

    # Build metadata
    metadata = {
        "document_id": doc["document_id"],
        "visibility": doc["visibility"],
        "owner_id": doc["owner_id"],
        "group_id": doc["group_id"],
        "source": doc["filename"],
        "content_hash": content_hash(content),
        "doc_domain": doc["doc_domain"],
    }
    if edited_at:
        metadata["edited"] = True
        metadata["edited_at"] = edited_at.isoformat() if hasattr(edited_at, "isoformat") else str(edited_at)

    point = PointStruct(
        id=chunk_id,
        vector=vector,
        payload={
            "page_content": content,
            "metadata": metadata,
        },
    )
    client.upsert(
        collection_name=settings.collection_name,
        points=[point],
    )


@backfill_app.command("run")
def backfill_run(
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run", help="Dry run mode"),
    document_id: int | None = typer.Option(None, "--document-id", help="Backfill specific document"),
) -> None:
    """Backfill chunk IDs to sync Postgres and Qdrant.

    This command re-indexes all documents so that Qdrant point IDs match
    Postgres chunk IDs. Required for chunk editing feature.
    """
    logger.info("Starting backfill of chunk IDs...")
    logger.info("Collection: %s", settings.collection_name)

    # Get all documents with chunks
    docs = asyncio.run(_get_documents_with_chunks(document_id))
    if not docs:
        logger.info("No documents found to backfill.")
        sys.exit(0)

    total_chunks = sum(len(d["chunks"]) for d in docs)
    logger.info("Found %d documents with %d total chunks", len(docs), total_chunks)

    if dry_run:
        logger.info("DRY RUN: Would process %d documents", len(docs))
        for doc in docs:
            logger.info(
                "  Document %d (%s): %d chunks",
                doc["document_id"],
                doc["filename"],
                len(doc["chunks"]),
            )
        logger.info("Run with --no-dry-run to execute backfill.")
        sys.exit(0)

    # Process each document
    t0 = time.monotonic()
    processed = 0
    errors = 0

    for doc in docs:
        doc_id = doc["document_id"]
        try:
            logger.info(
                "Processing document %d (%s): %d chunks...",
                doc_id,
                doc["filename"],
                len(doc["chunks"]),
            )

            # Delete existing points for this document
            _delete_document_points(doc_id)

            # Upsert each chunk with its Postgres ID
            for chunk in doc["chunks"]:
                _upsert_chunk(
                    chunk_id=chunk["id"],
                    content=chunk["content"],
                    doc=doc,
                    edited_at=chunk.get("edited_at"),
                )
                processed += 1

            logger.info("  Document %d: done", doc_id)

        except Exception as e:
            logger.error("  Document %d: error - %s", doc_id, e)
            errors += 1

    elapsed = time.monotonic() - t0
    logger.info(
        "Backfill completed in %.1fs: %d chunks processed, %d errors",
        elapsed,
        processed,
        errors,
    )

    if errors > 0:
        sys.exit(1)
    else:
        sys.exit(0)
