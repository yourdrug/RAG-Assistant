"""
infrastructure/qdrant_ops.py — Qdrant collection operations.
Extracted from vector_store.py. Pure functions receiving dependencies.
"""

import hashlib
import logging
import time
import uuid

from config import settings
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

log = logging.getLogger("default")


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def ensure_collection(client: QdrantClient, vector_size: int, reset: bool = False) -> None:
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
            return
    log.info("Creating collection '%s' (dim=%d) ...", settings.collection_name, vector_size)
    client.create_collection(
        collection_name=settings.collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


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
    hashes = [_content_hash(t) for t in texts]
    for doc, h in zip(chunks, hashes):
        doc.metadata["content_hash"] = h

    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct

    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)

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
