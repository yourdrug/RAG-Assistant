"""Qdrant collection operations -- pure functions receiving explicit dependencies.

Provides collection creation, point upsert/deletion, payload indexing,
and ACL-filtered search helpers.  Extracted from the original vector_store
module to keep concerns separated.
"""

import logging
import time
import uuid

from config import settings
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

from infrastructure.clients import get_qdrant_client
from infrastructure.ml.hybrid import content_hash

log = logging.getLogger("default")


def ensure_collection(client, vector_size: int, reset: bool = False) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if settings.collection_name in existing:
        if reset:
            log.info("Deleting collection '%s' ...", settings.collection_name)
            client.delete_collection(settings.collection_name)
        else:
            info = client.get_collection(settings.collection_name)
            count = info.points_count or 0
            log.info(
                "Collection '%s' exists — %d points. Adding new documents.",
                settings.collection_name,
                count,
            )
            _ensure_payload_indexes(client)
            return
    log.info("Creating collection '%s' (dim=%d) ...", settings.collection_name, vector_size)
    try:
        client.create_collection(
            collection_name=settings.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
    except Exception as e:
        if "already exists" in str(e):
            log.info(
                "Collection '%s' already exists (created by concurrent process)", settings.collection_name
            )
        else:
            raise
    _ensure_payload_indexes(client)


def _ensure_payload_indexes(client) -> None:
    """Create payload indexes on ACL fields for efficient filtered search."""
    acl_fields = [
        ("metadata.visibility", PayloadSchemaType.KEYWORD),
        ("metadata.owner_id", PayloadSchemaType.INTEGER),
        ("metadata.group_id", PayloadSchemaType.INTEGER),
    ]
    for field_name, field_type in acl_fields:
        try:
            client.create_payload_index(
                collection_name=settings.collection_name,
                field_name=field_name,
                field_schema=field_type,
            )
            log.info("Payload index created: %s", field_name)
        except Exception as e:
            if "already exists" in str(e):
                log.debug("Payload index already exists: %s", field_name)
            else:
                log.warning("Failed to create payload index %s: %s", field_name, e)


def upload_to_qdrant(chunks: list[Document], embeddings: HuggingFaceEmbeddings) -> None:
    embed_batch = settings.embed_batch_size
    qdrant_batch = 500
    total = len(chunks)
    log.info(
        "Uploading %d chunks to Qdrant (embed batch=%d, upsert batch=%d) ...",
        total,
        embed_batch,
        qdrant_batch,
    )
    t0 = time.monotonic()

    # Pre-compute all content hashes (faster than per-doc)
    texts = [doc.page_content for doc in chunks]
    hashes = [content_hash(t) for t in texts]
    for doc, h in zip(chunks, hashes):
        doc.metadata["content_hash"] = h

    from qdrant_client.models import PointStruct

    client = get_qdrant_client()

    # Embed + upsert in sub-batches to limit peak memory and give visible progress
    pending_points: list[PointStruct] = []
    for batch_start in range(0, total, embed_batch):
        batch_end = min(batch_start + embed_batch, total)
        batch_texts = texts[batch_start:batch_end]
        batch_chunks_slice = chunks[batch_start:batch_end]

        log.info("  Embedding chunks %d/%d ...", batch_end, total)
        t_embed = time.monotonic()
        batch_vectors = embeddings.embed_documents(batch_texts)
        log.info("  Embedded in %.1fs", time.monotonic() - t_embed)

        for doc, vector in zip(batch_chunks_slice, batch_vectors):
            pending_points.append(
                PointStruct(
                    id=uuid.uuid4().hex,
                    vector=vector,
                    payload={
                        "page_content": doc.page_content,
                        "metadata": doc.metadata,
                    },
                )
            )

        # Flush to Qdrant when we have enough for a qdrant_batch or at the end
        while len(pending_points) >= qdrant_batch:
            client.upsert(
                collection_name=settings.collection_name,
                points=pending_points[:qdrant_batch],
            )
            pending_points = pending_points[qdrant_batch:]

        done = batch_end
        elapsed = time.monotonic() - t0
        speed = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / speed if speed > 0 else 0
        log.info(
            "  Progress %d/%d chunks  (%.1f c/s, ETA ~%.0fs)",
            done,
            total,
            speed,
            eta,
        )

    # Flush remaining points
    if pending_points:
        client.upsert(
            collection_name=settings.collection_name,
            points=pending_points,
        )

    log.info("Qdrant upload completed in %.1fs", time.monotonic() - t0)
