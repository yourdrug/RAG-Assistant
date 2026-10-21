"""DocumentPipeline -- единый shared-слой для обработки чанков.

Используется и API (DocumentProcessor), и CLI (IngestionService).
Гарантирует одинаковое поведение: bulk_insert → outbox enqueue → status update.
"""

from __future__ import annotations

from datetime import date, datetime
import logging
from typing import TYPE_CHECKING, Unpack

from application.indexing.classification import classify_document_text
from domain.knowledge.chunk import ChunkInsert, extract_chunk_context
from domain.knowledge.enums import DocumentStatus, DocumentVisibility
from domain.knowledge.index_sync import OutboxOperation, VectorOutboxEntry
from domain.shared.hashing import content_hash


if TYPE_CHECKING:
    from application.indexing.dto import ChunkProcessingOptions, ChunkWriteOptions, VersionMetadataOptions
    from application.uow import UnitOfWork, UnitOfWorkFactory
    from domain.ingestion.ports import DomainSettingsPort
    from domain.ingestion.profiles.registry import DomainProfileRegistry
    from domain.ingestion.raw_document import RawDocument

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
        "source": filename,
        "doc_domain": doc_domain,
    }
    base.update(extra)
    return base


def build_version_metadata(versioning: VersionMetadataOptions) -> dict[str, object]:
    act_version_id = versioning.get("act_version_id")
    act_id = versioning.get("act_id")
    effective_from = versioning.get("effective_from")
    effective_to = versioning.get("effective_to")
    temporal_metadata: dict[str, object] = {
        key: value
        for key, value in (("act_version_id", act_version_id), ("act_id", act_id))
        if value is not None
    }

    for key, temporal in (("effective_from", effective_from), ("effective_to", effective_to)):
        if temporal is not None:
            temporal_metadata[key] = temporal.isoformat() if isinstance(temporal, date) else temporal

    return temporal_metadata


def enrich_chunks_metadata(
    chunks: list[RawDocument],
    document_id: int,
    visibility: str,
    owner_id: int | None,
    group_id: int | None,
    *,
    doc_domain: str,
    **versioning: Unpack[VersionMetadataOptions],
) -> None:
    """Обогащает metadata каждого чанка стандартными полями.

    Вызывать ПОСЛЕ split и ПЕРЕД process_chunks.
    """
    domain_metadata = versioning.get("domain_metadata")
    temporal_metadata = build_version_metadata(versioning)

    for rc in chunks:
        meta = {
            "document_id": document_id,
            "visibility": visibility,
            "owner_id": owner_id,
            "group_id": group_id,
            "doc_domain": doc_domain,
            "is_current": versioning.get("is_current", True),
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

        meta.update(temporal_metadata)
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
    }

    if existing_uow is not None:
        await process_chunks_in_uow(existing_uow, document_id, filename, chunks, visibility, **write_options)
        return

    async with uow_factory.create(master=True) as uow:
        await process_chunks_in_uow(uow, document_id, filename, chunks, visibility, **write_options)


def parse_meta_datetime(value: object) -> date | None:
    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, date) or value is None:
        return value

    try:
        parsed = date.fromisoformat(str(value))
    except ValueError:
        try:
            parsed = datetime.fromisoformat(str(value)).date()
        except ValueError:
            parsed = None
    return parsed


async def process_chunks_in_uow(
    uow: UnitOfWork,
    document_id: int,
    filename: str,
    chunks: list[RawDocument],
    visibility: str,
    **options: Unpack[ChunkWriteOptions],
) -> None:
    replaced_ids = await uow.chunks.get_ids_by_document_id(document_id, for_update=True)
    # 1. Bulk insert → получаем chunk_ids
    # Doc-level versioning fields live identically on every chunk's metadata.
    first_meta = chunks[0].metadata if chunks else {}
    chunk_ids = await uow.chunks.bulk_insert(
        document_id=document_id,
        filename=filename,
        visibility=visibility,
        chunks=[
            ChunkInsert(
                content=rc.page_content,
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
        ],
        owner_id=options["owner_id"],
        group_id=options["group_id"],
        doc_domain=options["doc_domain"],
        domain_metadata=first_meta.get("domain_metadata"),
        act_version_id=first_meta.get("act_version_id"),
        effective_from=parse_meta_datetime(first_meta.get("effective_from")),
        effective_to=parse_meta_datetime(first_meta.get("effective_to")),
        is_current=first_meta.get("is_current", True),
    )

    # 2. Enrich metadata with chunk_ids и chunk_index
    for i, (chunk_id, rc) in enumerate(zip(chunk_ids, chunks, strict=True)):
        rc.metadata["chunk_id"] = chunk_id
        rc.metadata["chunk_index"] = i

    # 3. Enqueue outbox for Qdrant
    if replaced_ids:
        await uow.vector_outbox.enqueue(
            VectorOutboxEntry(
                operation=OutboxOperation.DELETE_CHUNKS,
                aggregate_type="document",
                aggregate_id=document_id,
                payload={"chunk_ids": replaced_ids},
            )
        )

    await uow.vector_outbox.enqueue(
        VectorOutboxEntry(
            operation=OutboxOperation.UPSERT_CHUNKS,
            aggregate_type="document",
            aggregate_id=document_id,
            payload={
                "points": [
                    {
                        "chunk_id": cid,
                        "page_content": rc.page_content,
                        "metadata": rc.metadata,
                    }
                    for cid, rc in zip(chunk_ids, chunks, strict=True)
                ]
            },
        )
    )

    # 4. Status update
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
        "Pipeline: doc %d -- %d chunks enqueued (%d chars)",
        document_id,
        len(chunks),
        total_chars,
    )
