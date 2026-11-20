"""Qdrant collection operations -- pure functions receiving explicit dependencies.

Provides collection creation, point upsert/deletion, payload indexing,
and ACL-filtered search helpers.  Extracted from the original vector_store
module to keep concerns separated.
"""

import asyncio
from collections.abc import Awaitable, Callable
import logging
import time

from langchain.schema import Document
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    IsNullCondition,
    MatchValue,
    Modifier,
    PayloadField,
    PayloadSchemaType,
    PointStruct,
    SparseVectorParams,
    VectorParams,
)

from application.ports.ml_clients import AsyncSemaphorePort
from config import MLProviderSection, process_section
from domain.knowledge.revision import EMBED_HASH_KEY, REUSE_POINT_KEY
from domain.shared.hashing import content_hash
from infrastructure.bm25.document_text import bm25_document_text
from infrastructure.corpus.snapshot import collection_for
from infrastructure.ml.clients.client_lease import client_operation
from infrastructure.ml.clients.factories import create_qdrant_client
from infrastructure.repositories.vector.client_types import DocumentEmbeddingClient, VectorClient
from infrastructure.repositories.vector.embedding_identity import ensure_embedding_identity
from infrastructure.repositories.vector.sparse_vectors import SPARSE_VECTOR_NAME, point_vectors


# Buffered points carry vectors plus ~10 KB of chunk metadata each; a small
# buffer bounds peak memory without slowing writes measurably.
QDRANT_UPSERT_BATCH_SIZE = 64
# Upload inputs that are not part of the stored point payload.
UPLOAD_ONLY_METADATA = frozenset({"embed_text", REUSE_POINT_KEY})


log = logging.getLogger("default")


def validate_embedding_batch(
    vectors: list[list[float]], expected_count: int, expected_dimension: int
) -> None:
    if len(vectors) != expected_count:
        msg = (
            "Embedding provider returned an incomplete batch "
            f"({len(vectors)} vectors for {expected_count} chunks)"
        )
        raise ValueError(msg)

    for vector_index, vector in enumerate(vectors):
        if len(vector) != expected_dimension:
            msg = (
                "Embedding provider returned an invalid vector dimension "
                f"at batch index {vector_index}: expected {expected_dimension}, got {len(vector)}"
            )
            raise ValueError(msg)


def delete_internal_points(client: VectorClient) -> int:
    """Delete CLI-ingested and public UI points, but preserve manual document chunks.

    Deletes points where owner_id=None AND manual != True.
    Returns the number of points deleted.
    """
    info = client.get_collection(collection_for(client))
    total_before = info.points_count or 0

    client.delete(
        collection_name=collection_for(client),
        points_selector=Filter(
            must=[
                IsNullCondition(
                    is_null=PayloadField(key="metadata.owner_id"),
                )
            ],
            must_not=[
                FieldCondition(
                    key="metadata.manual",
                    match=MatchValue(value=True),
                )
            ],
        ),
    )

    info_after = client.get_collection(collection_for(client))
    total_after = info_after.points_count or 0
    return total_before - total_after


def ensure_collection(client: VectorClient, vector_size: int, *, reset: bool = False) -> None:
    ensure_embedding_identity(client)
    existing = [c.name for c in client.get_collections().collections]

    if collection_for(client) in existing:
        info = client.get_collection(collection_for(client))
        # Validate vector dimension matches expected size
        vectors_config = info.config.params.vectors
        existing_dim = getattr(vectors_config, "size", None)

        if existing_dim is not None and existing_dim != vector_size:
            msg = (
                f"Collection '{collection_for(client)}' has vector dimension {existing_dim}, "
                f"but the current embedding model requires {vector_size}. "
                "To migrate: create a new collection with the correct dimension and re-ingest documents."
            )
            raise ValueError(msg)

        sparse = info.config.params.sparse_vectors or {}

        if SPARSE_VECTOR_NAME not in sparse or sparse[SPARSE_VECTOR_NAME].modifier != Modifier.IDF:
            msg = (
                "Collection has no compatible BM25 sparse index; "
                "run `corpus migrate-sparse` (task corpus:migrate-sparse)"
            )
            raise ValueError(msg)

        if reset:
            log.info(
                "Smart reset: deleting CLI-ingested points (owner_id=None) from collection '%s' ...",
                collection_for(client),
            )
            deleted = delete_internal_points(client)
            log.info("Deleted %d internal points. API-uploaded points preserved.", deleted)
        else:
            count = info.points_count or 0
            log.info(
                "Collection '%s' exists — %d points (dim=%d). Adding new documents.",
                collection_for(client),
                count,
                existing_dim,
            )

        ensure_payload_indexes(client)
        return

    log.info("Creating collection '%s' (dim=%d) ...", collection_for(client), vector_size)
    try:
        client.create_collection(
            collection_name=collection_for(client),
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            sparse_vectors_config={SPARSE_VECTOR_NAME: SparseVectorParams(modifier=Modifier.IDF)},
        )
    except Exception as e:
        if "already exists" in str(e):
            log.info("Collection '%s' already exists (created by concurrent process)", collection_for(client))
        else:
            raise
    ensure_payload_indexes(client)


def ensure_payload_indexes(client: VectorClient) -> None:
    """Create payload indexes on ACL and content fields for efficient filtered search."""
    acl_fields = [
        ("metadata.visibility", PayloadSchemaType.KEYWORD),
        ("metadata.owner_id", PayloadSchemaType.INTEGER),
        ("metadata.group_id", PayloadSchemaType.INTEGER),
        ("metadata.space_id", PayloadSchemaType.INTEGER),
        ("metadata.doc_domain", PayloadSchemaType.KEYWORD),
        ("metadata.document_id", PayloadSchemaType.INTEGER),
        ("metadata.content_hash", PayloadSchemaType.KEYWORD),
        ("metadata.content_type", PayloadSchemaType.KEYWORD),
        ("metadata.doc_type", PayloadSchemaType.KEYWORD),
        ("metadata.has_dates", PayloadSchemaType.BOOL),
        # Every search excludes points of unpublished revisions.
        ("metadata.staged", PayloadSchemaType.BOOL),
    ]

    for field_name, field_type in acl_fields:
        try:
            client.create_payload_index(
                collection_name=collection_for(client),
                field_name=field_name,
                field_schema=field_type,
            )
            log.info("Payload index created: %s", field_name)
        except Exception as e:
            if "already exists" in str(e):
                log.debug("Payload index already exists: %s", field_name)
            else:
                log.warning("Failed to create payload index %s: %s", field_name, e, exc_info=True)


async def upsert_with_semaphore(
    client: VectorClient, points: list[PointStruct], write_semaphore: AsyncSemaphorePort | None = None
) -> None:
    """Upsert points to Qdrant, optionally gated by a write semaphore."""
    if write_semaphore is not None:
        async with write_semaphore:
            await asyncio.to_thread(
                client.upsert,
                collection_name=collection_for(client),
                points=points,
            )
    else:
        await asyncio.to_thread(
            client.upsert,
            collection_name=collection_for(client),
            points=points,
        )


def tag_embedding_model(chunks: list[Document], identity: str) -> list[Document]:
    for doc in chunks:
        doc.metadata["embedding_model"] = identity

    return chunks


def build_points(chunks: list[Document], vectors: list[list[float]]) -> list[PointStruct]:
    """Encode sparse vectors off the event loop alongside the supplied dense vectors."""
    return [
        PointStruct(
            id=doc.metadata["chunk_id"],
            vector=point_vectors(bm25_document_text(doc), vector),
            payload={
                "page_content": doc.page_content,
                "metadata": {
                    key: value for key, value in doc.metadata.items() if key not in UPLOAD_ONLY_METADATA
                },
            },
        )
        for doc, vector in zip(chunks, vectors, strict=True)
    ]


async def upload_to_qdrant(
    chunks: list[Document],
    embeddings: DocumentEmbeddingClient,
    client: VectorClient | None = None,
    *,
    write_semaphore: AsyncSemaphorePort | None = None,
    should_cancel: Callable[[], Awaitable[bool]] | None = None,
) -> None:
    async with client_operation(embeddings):
        await upload_document_batches(
            chunks, embeddings, client=client, write_semaphore=write_semaphore, should_cancel=should_cancel
        )


def filter_upload_chunks(chunks: list[Document]) -> list[Document]:
    # Filter out empty/whitespace-only chunks that would cause TEI errors
    original_count = len(chunks)
    chunks = [c for c in chunks if c.page_content.strip()]

    if len(chunks) < original_count:
        log.warning(
            "Filtered %d empty/whitespace chunks → %d remaining",
            original_count - len(chunks),
            len(chunks),
        )

    return chunks


def prepare_upload_texts(chunks: list[Document]) -> list[str]:
    # Identity is always the full persisted content; dense embeddings keep full scope.
    texts = [doc.metadata.get("embed_text") or doc.page_content for doc in chunks]

    for doc, text in zip(chunks, texts, strict=True):
        doc.metadata["content_hash"] = content_hash(doc.page_content)
        # Proves later that this point's dense vector encodes exactly this input.
        doc.metadata[EMBED_HASH_KEY] = content_hash(text)

    return texts


def dense_part(vector: object) -> list[float] | None:
    if isinstance(vector, dict):
        vector = vector.get("")

    return vector if isinstance(vector, list) else None


def reusable_vectors(client: VectorClient, chunks: list[Document], identity: str) -> dict[int, list[float]]:
    """Dense vectors of existing points that encode the same input with the same model.

    Keyed by chunk position. A point qualifies only if its stored input hash and
    embedding model match, so a stale or foreign point is never reused.
    """
    wanted = {
        position: point_id
        for position, doc in enumerate(chunks)
        if isinstance(point_id := doc.metadata.get(REUSE_POINT_KEY), int)
    }

    if not wanted:
        return {}

    records = client.retrieve(
        collection_name=collection_for(client),
        ids=sorted(set(wanted.values())),
        with_payload=True,
        with_vectors=True,
    )
    by_id = {record.id: record for record in records}
    reused: dict[int, list[float]] = {}

    for position, point_id in wanted.items():
        record = by_id.get(point_id)
        metadata = (record.payload or {}).get("metadata") if record is not None else None
        dense = dense_part(record.vector) if record is not None else None

        if (
            isinstance(metadata, dict)
            and dense is not None
            and metadata.get("embedding_model") == identity
            and metadata.get(EMBED_HASH_KEY) == chunks[position].metadata[EMBED_HASH_KEY]
        ):
            reused[position] = dense

    return reused


async def tag_and_find_reusable(
    client: VectorClient, chunks: list[Document], embeddings: DocumentEmbeddingClient
) -> dict[int, list[float]]:
    """Tag chunks with the corpus embedding model; return vectors proven reusable for them."""
    identity = await asyncio.to_thread(
        ensure_embedding_identity, client, getattr(embeddings, "embedding_identity", None)
    )
    tag_embedding_model(chunks, identity)
    reused = await asyncio.to_thread(reusable_vectors, client, chunks, identity)

    if reused:
        log.info("Reusing %d/%d dense vectors of unchanged chunks", len(reused), len(chunks))

    return reused


async def embed_or_reuse(
    embeddings: DocumentEmbeddingClient, texts: list[str], reused: dict[int, list[float]], offset: int
) -> list[list[float]]:
    """Vectors for one batch: reused where proven identical, embedded otherwise."""
    missing = [index for index in range(len(texts)) if offset + index not in reused]
    embedded = await embeddings.embed_documents([texts[index] for index in missing]) if missing else []

    if len(embedded) != len(missing):
        message = f"Embedding count mismatch: expected {len(missing)}, got {len(embedded)}"
        raise ValueError(message)

    vectors = dict(zip(missing, embedded, strict=True))
    return [reused[offset + index] if index not in vectors else vectors[index] for index in range(len(texts))]


def log_upload_progress(done: int, total: int, t0: float) -> None:
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


async def upload_document_batches(
    chunks: list[Document],
    embeddings: DocumentEmbeddingClient,
    client: VectorClient | None = None,
    *,
    write_semaphore: AsyncSemaphorePort | None = None,
    should_cancel: Callable[[], Awaitable[bool]] | None = None,
) -> None:
    embed_batch = process_section(MLProviderSection).embed_batch_size

    chunks = filter_upload_chunks(chunks)

    if not chunks:
        log.warning("No non-empty chunks to upload — skipping Qdrant upload")
        return

    total = len(chunks)
    log.info(
        "Uploading %d chunks to Qdrant (embed batch=%d, upsert batch=%d) ...",
        total,
        embed_batch,
        QDRANT_UPSERT_BATCH_SIZE,
    )
    t0 = time.monotonic()

    texts = prepare_upload_texts(chunks)

    # Reuse the caller's client when provided (avoids a new TCP pool per call)
    if client is None:
        client = create_qdrant_client()

    reused = await tag_and_find_reusable(client, chunks, embeddings)

    # Embed + upsert in sub-batches to limit peak memory and give visible progress
    pending_points: list[PointStruct] = []

    for batch_start in range(0, total, embed_batch):
        batch_end = min(batch_start + embed_batch, total)
        batch_texts = texts[batch_start:batch_end]
        batch_chunks_slice = chunks[batch_start:batch_end]

        if should_cancel is not None and await should_cancel():
            log.info(
                "Upload cancelled at %d/%d chunks — document deleted during embedding",
                batch_start,
                total,
            )
            return

        log.info("  Embedding chunks %d/%d ...", batch_end, total)
        t_embed = time.monotonic()
        batch_vectors = await embed_or_reuse(embeddings, batch_texts, reused, batch_start)
        log.info("  Embedded in %.1fs", time.monotonic() - t_embed)

        validate_embedding_batch(
            batch_vectors, len(batch_chunks_slice), process_section(MLProviderSection).embed_dim
        )

        pending_points.extend(await asyncio.to_thread(build_points, batch_chunks_slice, batch_vectors))

        # Flush to Qdrant when we have enough for a QDRANT_UPSERT_BATCH_SIZE or at the end
        while len(pending_points) >= QDRANT_UPSERT_BATCH_SIZE:
            await upsert_with_semaphore(client, pending_points[:QDRANT_UPSERT_BATCH_SIZE], write_semaphore)
            pending_points = pending_points[QDRANT_UPSERT_BATCH_SIZE:]

        log_upload_progress(batch_end, total, t0)

    # Flush remaining points
    if pending_points:
        await upsert_with_semaphore(client, pending_points, write_semaphore)

    log.info("Qdrant upload completed in %.1fs", time.monotonic() - t0)
