"""DocumentPipeline -- единый shared-слой для обработки чанков.

Используется и API (DocumentProcessor), и CLI (IngestionService).
Гарантирует одинаковое поведение: bulk_insert → outbox enqueue → status update.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Unpack

from application.indexing.classification import classify_document_text
from domain.knowledge.chunk import ChunkInsert, extract_chunk_context
from domain.knowledge.document_nodes import NODE_PATH_KEY, build_document_nodes
from domain.knowledge.enums import DocumentStatus, DocumentVisibility
from domain.knowledge.index_sync import OutboxOperation, VectorOutboxEntry
from domain.knowledge.pack_metadata import PACK_METADATA_KEY
from domain.knowledge.repositories.chunk import ChunkInsertOptions
from domain.knowledge.revision import (
    REUSE_POINT_KEY,
    STAGED_REVISION_KEY,
    embedding_input,
    match_reusable_chunks,
)
from domain.knowledge.space import SPACE_ID_KEY
from domain.shared.hashing import content_hash


if TYPE_CHECKING:
    from application.indexing.dto import ChunkProcessingOptions, ChunkWriteOptions, DocumentMetadataOptions
    from application.uow import UnitOfWork, UnitOfWorkFactory
    from domain.ingestion.ports import DomainSettingsPort
    from domain.ingestion.profiles.registry import DomainProfileRegistry
    from domain.ingestion.raw_document import RawDocument
    from domain.knowledge.chunk import ChunkRecord
    from domain.knowledge.document_nodes import DocumentNode
    from domain.knowledge.revision import SourceVersion, StagedRevision

log = logging.getLogger("default")


def tag_chunks(
    chunks: list[RawDocument],
    *,
    visibility: DocumentVisibility = DocumentVisibility.INTERNAL_PUBLIC,
    owner_id: int | None = None,
    group_id: int | None = None,
    client_id: int | None = None,
) -> None:
    for c in chunks:
        c.metadata.update(
            {
                "visibility": visibility.value,
                "owner_id": owner_id,
                "group_id": group_id,
                "client_id": client_id,
            }
        )


def tag_domain(chunks: list[RawDocument], doc_domain: str) -> None:
    for c in chunks:
        c.metadata["doc_domain"] = doc_domain


def classify_domain(
    text: str,
    *,
    domain_registry: DomainProfileRegistry | None = None,
    domain_settings: DomainSettingsPort | None = None,
) -> str:
    """Shared registry-based domain classification.

    Used by both IngestionService (CLI) and DocumentProcessor (API) -- the
    decision logic lives in indexing.classification.classify_document_text so
    API and CLI can never drift apart.  CLI discards the ambiguity warning
    (it has no warning_message channel on this path).
    """
    return classify_document_text(
        text,
        domain_registry=domain_registry,
        domain_settings=domain_settings,
    ).domain_key


def build_outbox_metadata(
    document_id: int,
    visibility: DocumentVisibility | str,
    owner_id: int | None,
    group_id: int | None,
    *,
    space_id: int | None,
    filename: str,
    doc_domain: str,
    **extra: object,
) -> dict[str, object]:
    """Build a standard metadata dict for vector outbox entries.

    Centralizes the metadata construction to avoid duplication across
    chunk_service, document_service, and document_pipeline.
    """
    vis = visibility.value if isinstance(visibility, DocumentVisibility) else visibility
    base = {
        "document_id": document_id,
        "visibility": vis,
        "owner_id": owner_id,
        "group_id": group_id,
        SPACE_ID_KEY: space_id,
        "source": filename,
        "doc_domain": doc_domain,
    }
    base.update(extra)
    return base


def enrich_chunks_metadata(
    chunks: list[RawDocument],
    document_id: int,
    visibility: str,
    owner_id: int | None,
    group_id: int | None,
    *,
    doc_domain: str,
    **document_metadata: Unpack[DocumentMetadataOptions],
) -> None:
    """Обогащает metadata каждого чанка стандартными полями.

    Вызывать ПОСЛЕ split и ПЕРЕД process_chunks.
    """
    domain_metadata = document_metadata.get("domain_metadata")
    pack_metadata = dict(document_metadata.get("pack_metadata") or {})

    for rc in chunks:
        meta = {
            "document_id": document_id,
            "visibility": visibility,
            "owner_id": owner_id,
            "group_id": group_id,
            "doc_domain": doc_domain,
        }

        if domain_metadata:
            # Chunk-level refs (set by content-based splitting) take precedence;
            # doc-level refs (decree_number, decree_date, ...) fill the gaps so
            # every chunk of a versioned document carries the document identity.
            existing = rc.metadata.get("domain_metadata") or {}
            merged = dict(existing)

            for key, value in domain_metadata.items():
                if key not in merged:
                    merged[key] = value

            if merged:
                meta["domain_metadata"] = merged

        if pack_metadata:
            meta[PACK_METADATA_KEY] = dict(pack_metadata)

        rc.metadata.update(meta)


async def enqueue_delete_by_document(uow: UnitOfWork, document_id: int) -> None:
    """Enqueue a DELETE_BY_DOCUMENT outbox entry.

    Shared by the sync conflict path (document_conflict_resolver) and the
    async persist path -- both must remove the replaced document from Qdrant.
    """
    await uow.vector_outbox.enqueue(
        VectorOutboxEntry(
            operation=OutboxOperation.DELETE_BY_DOCUMENT,
            aggregate_type="document",
            aggregate_id=document_id,
            payload={"document_id": document_id},
        )
    )


async def process_chunks(
    uow_factory: UnitOfWorkFactory,
    document_id: int,
    filename: str,
    chunks: list[RawDocument],
    visibility: str,
    **options: Unpack[ChunkProcessingOptions],
) -> None:
    """Persist in the caller's transaction or open an owned transaction."""
    existing_uow = options.get("_existing_uow")
    write_options: ChunkWriteOptions = {
        "owner_id": options["owner_id"],
        "group_id": options["group_id"],
        "doc_domain": options["doc_domain"],
        "set_indexing": options.get("set_indexing", True),
        "source_version": options.get("source_version"),
    }

    if existing_uow is not None:
        await process_chunks_in_uow(existing_uow, document_id, filename, chunks, visibility, **write_options)
        return

    async with uow_factory.create(master=True) as uow:
        await process_chunks_in_uow(uow, document_id, filename, chunks, visibility, **write_options)


def chunk_inserts(chunks: list[RawDocument]) -> list[ChunkInsert]:
    return [
        ChunkInsert(
            content=rc.page_content,
            embed_text=rc.embed_text,
            content_hash=content_hash(rc.page_content),
            section=rc.metadata.get("section"),
            heading=rc.metadata.get("heading"),
            heading_level=rc.metadata.get("heading_level"),
            content_type=rc.metadata.get("content_type"),
            doc_title=rc.metadata.get("doc_title"),
            doc_type=rc.metadata.get("doc_type"),
            context_metadata=extract_chunk_context(rc.metadata),
        )
        for rc in chunks
    ]


def reusable_point_ids(published: list[ChunkRecord], chunks: list[RawDocument]) -> list[int | None]:
    """Published chunks whose dense vectors may serve the new chunks (same dense input)."""
    return match_reusable_chunks(
        [(chunk.chunk_id, embedding_input(chunk.content, chunk.embed_text)) for chunk in published],
        [embedding_input(rc.page_content, rc.embed_text) for rc in chunks],
    )


async def process_chunks_in_uow(
    uow: UnitOfWork,
    document_id: int,
    filename: str,
    chunks: list[RawDocument],
    visibility: str,
    **options: Unpack[ChunkWriteOptions],
) -> None:
    """Write a document's chunks and enqueue their vector upload.

    A document without published chunks is written directly. Otherwise the new
    chunks become a staged revision: search keeps the published one until the
    dispatcher has indexed every staged point and publishes the revision.
    A changed ACL replaces the chunks at once instead: narrowed access must
    apply to the old content immediately, in SQL search as well.
    """
    source_version = options.get("source_version")
    published_ids = await uow.chunks.get_ids_by_document_id(document_id, for_update=True)
    document = await uow.documents.get_by_id(document_id)
    # Chunks and their vector payload carry the document's space as a search selection label.
    space_id = document.space_id if document is not None else None
    insert_options = chunk_insert_options(chunks, space_id, options)
    # A newer processing run supersedes an unpublished revision; its points go too.
    superseded_ids = await uow.chunks.discard_staged_revision(document_id)

    if superseded_ids:
        await enqueue_delete_chunks(uow, document_id, superseded_ids)

    staged: StagedRevision | None = None
    reuse: list[int | None] = [None] * len(chunks)
    published = await uow.chunks.list_document_chunks([document_id]) if published_ids and chunks else []

    if published and not acl_changed(published, visibility, options["owner_id"], options["group_id"]):
        reuse = reusable_point_ids(published, chunks)
        staged = await uow.chunks.stage_revision(
            document_id,
            filename,
            visibility,
            chunk_inserts(chunks),
            nodes=document_nodes(chunks),
            options=insert_options,
            source_version=source_version,
        )
        chunk_ids = list(staged.chunk_ids)
    else:
        chunk_ids = await write_published_chunks(
            uow,
            document_id,
            filename,
            visibility,
            chunks,
            options=insert_options,
            source_version=source_version,
        )

        if published_ids:
            # Re-processing produced no chunks: nothing to publish, the old points go.
            await enqueue_delete_chunks(uow, document_id, published_ids)

    for i, (chunk_id, rc) in enumerate(zip(chunk_ids, chunks, strict=True)):
        rc.metadata["chunk_id"] = chunk_id
        rc.metadata["chunk_index"] = i
        rc.metadata[SPACE_ID_KEY] = space_id
        rc.metadata.pop(NODE_PATH_KEY, None)

    await uow.vector_outbox.enqueue(
        VectorOutboxEntry(
            operation=OutboxOperation.UPSERT_CHUNKS,
            aggregate_type="document",
            aggregate_id=document_id,
            payload=upsert_payload(chunk_ids, chunks, reuse, staged),
        )
    )

    total_chars = sum(len(rc.page_content) for rc in chunks)
    await uow.documents.update_status(
        document_id,
        status=DocumentStatus.INDEXING.value
        if options.get("set_indexing", True)
        else DocumentStatus.DONE.value,
        chunks=len(chunks),
        chars=total_chars,
    )

    log.info(
        "Pipeline: doc %d -- %d chunks enqueued (%d chars, staged=%s, reusable vectors=%d)",
        document_id,
        len(chunks),
        total_chars,
        staged is not None,
        sum(item is not None for item in reuse),
    )


def document_nodes(chunks: list[RawDocument]) -> list[DocumentNode]:
    return build_document_nodes([(rc.page_content, rc.metadata) for rc in chunks])


async def write_published_chunks(
    uow: UnitOfWork,
    document_id: int,
    filename: str,
    visibility: str,
    chunks: list[RawDocument],
    *,
    options: ChunkInsertOptions,
    source_version: SourceVersion | None,
) -> list[int]:
    """Replace the published chunks directly; they serve SQL search at once."""
    chunk_ids = await uow.chunks.bulk_insert(
        document_id=document_id,
        filename=filename,
        visibility=visibility,
        chunks=chunk_inserts(chunks),
        options=options,
    )
    # Parent scopes are replaced with the chunks: their ranges are these chunk indexes.
    await uow.chunks.replace_document_nodes(document_id, document_nodes(chunks))
    # A staged revision changes the domain only when published (see the dispatcher).
    await uow.documents.set_domain(document_id, options.doc_domain)

    if source_version is not None:
        await uow.documents.set_storage_version(document_id, source_version.version, source_version.revision)

    return chunk_ids


def chunk_insert_options(
    chunks: list[RawDocument], space_id: int | None, options: ChunkWriteOptions
) -> ChunkInsertOptions:
    # Doc-level domain and pack metadata live identically on every chunk's metadata.
    first_meta = chunks[0].metadata if chunks else {}
    return ChunkInsertOptions(
        owner_id=options["owner_id"],
        group_id=options["group_id"],
        space_id=space_id,
        doc_domain=options["doc_domain"],
        domain_metadata=first_meta.get("domain_metadata"),
        pack_metadata=dict(first_meta.get(PACK_METADATA_KEY) or {}),
    )


def upsert_payload(
    chunk_ids: list[int], chunks: list[RawDocument], reuse: list[int | None], staged: StagedRevision | None
) -> dict[str, object]:
    payload: dict[str, object] = {
        "points": [
            {
                "chunk_id": cid,
                "page_content": rc.page_content,
                "embed_text": rc.embed_text,
                "metadata": rc.metadata,
                **({REUSE_POINT_KEY: reuse_id} if reuse_id is not None else {}),
            }
            for cid, rc, reuse_id in zip(chunk_ids, chunks, reuse, strict=True)
        ]
    }

    if staged is not None:
        payload[STAGED_REVISION_KEY] = staged.token

    return payload


def acl_changed(
    published: list[ChunkRecord], visibility: str, owner_id: int | None, group_id: int | None
) -> bool:
    return any(
        (str(chunk.visibility), chunk.owner_id, chunk.group_id) != (str(visibility), owner_id, group_id)
        for chunk in published
    )


async def enqueue_delete_chunks(uow: UnitOfWork, document_id: int, chunk_ids: list[int]) -> None:
    await uow.vector_outbox.enqueue(
        VectorOutboxEntry(
            operation=OutboxOperation.DELETE_CHUNKS,
            aggregate_type="document",
            aggregate_id=document_id,
            payload={"chunk_ids": chunk_ids},
        )
    )
