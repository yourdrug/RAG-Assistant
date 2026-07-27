"""
infrastructure/qdrant_ops.py — Qdrant collection operations.
Extracted from vector_store.py. Pure functions receiving dependencies.
"""

import hashlib
import logging
import time

from config import settings
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
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
    batch_size = 500
    total = len(chunks)
    log.info("Uploading %d chunks to Qdrant in batches of %d ...", total, batch_size)
    t0 = time.monotonic()

    # Pre-compute all content hashes in batch (faster than per-doc)
    texts = [doc.page_content for doc in chunks]
    hashes = [_content_hash(t) for t in texts]
    for doc, h in zip(chunks, hashes):
        doc.metadata["content_hash"] = h

    # Pre-compute all embeddings in batch (major speedup vs per-query)
    log.info("Generating embeddings for %d chunks ...", total)
    t_embed = time.monotonic()
    embeddings_list = embeddings.embed_documents(texts)
    log.info("Embeddings generated in %.1fs", time.monotonic() - t_embed)

    # Upload to Qdrant with pre-computed vectors
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct

    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)

    for i in range(0, total, batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_vectors = embeddings_list[i : i + batch_size]

        points = []
        for j, (doc, vector) in enumerate(zip(batch_chunks, batch_vectors)):
            point_id = hashes[i + j]  # Use content_hash as ID for dedup
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "page_content": doc.page_content,
                        "metadata": doc.metadata,
                    },
                )
            )

        client.upsert(
            collection_name=settings.collection_name,
            points=points,
        )

        done = min(i + batch_size, total)
        elapsed = time.monotonic() - t0
        speed = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / speed if speed > 0 else 0
        log.info(
            "  Uploaded %d/%d chunks  (%.1f c/s, ETA ~%.0fs)",
            done,
            total,
            speed,
            eta,
        )

    log.info("Qdrant upload completed in %.1fs", time.monotonic() - t0)
