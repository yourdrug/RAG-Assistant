"""DocumentPipeline — единый shared-слой для обработки чанков.

Используется и API (DocumentProcessor), и CLI (IngestionService).
Гарантирует одинаковое поведение: bulk_insert → outbox enqueue → status update.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from domain.entities.vector_outbox_entry import OutboxOperation, VectorOutboxEntry
from domain.utils import content_hash
from domain.value_objects.document_status import DocumentStatus
from domain.value_objects.visibility import DocumentVisibility

if TYPE_CHECKING:
    from application.ports.unit_of_work_factory import UnitOfWorkFactory

log = logging.getLogger("default")


def build_outbox_metadata(
    document_id: int,
    visibility,
    owner_id: int | None,
    group_id: int | None,
    filename: str,
    doc_domain: str,
    **extra,
) -> dict:
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


def enrich_chunks_metadata(
    chunks: list[Any],
    document_id: int,
    visibility: str,
    owner_id: int | None,
    group_id: int | None,
    doc_domain: str,
) -> None:
    """Обогащает metadata каждого чанка стандартными полями.

    Вызывать ПОСЛЕ split и ПЕРЕД process_chunks.
    """
    for rc in chunks:
        rc.metadata.update(
            {
                "document_id": document_id,
                "visibility": visibility,
                "owner_id": owner_id,
                "group_id": group_id,
                "doc_domain": doc_domain,
            }
        )


async def process_chunks(
    uow_factory: UnitOfWorkFactory,
    document_id: int,
    filename: str,
    chunks: list[Any],
    visibility: str,
    owner_id: int | None,
    group_id: int | None,
    doc_domain: str,
    *,
    set_indexing: bool = True,
    _existing_uow: Any | None = None,
) -> None:
    """Единый pipeline: bulk_insert → outbox enqueue → status update.

    Args:
        uow_factory: фабрика UoW.
        document_id: ID документа в Postgres.
        filename: имя файла.
        chunks: список RawChunk с page_content и metadata.
        visibility: видимость документа.
        owner_id: ID владельца.
        group_id: ID группы.
        doc_domain: домен документа (general/legal).
        set_indexing: ставить ли статус indexing (True для API, False для CLI).
        _existing_uow: если передан — используется вместо создания нового
                       (нужно, чтобы chunks попали в ту же транзакцию, что и документ).

    """
    hashes = [content_hash(rc.page_content) for rc in chunks]

    if _existing_uow is not None:
        uow = _existing_uow
        await _process_chunks_in_uow(
            uow, document_id, filename, chunks, visibility,
            owner_id, group_id, doc_domain, hashes, set_indexing,
        )
        return

    async with uow_factory.create(master=True) as uow:
        await _process_chunks_in_uow(
            uow, document_id, filename, chunks, visibility,
            owner_id, group_id, doc_domain, hashes, set_indexing,
        )


async def _process_chunks_in_uow(
    uow: Any,
    document_id: int,
    filename: str,
    chunks: list[Any],
    visibility: str,
    owner_id: int | None,
    group_id: int | None,
    doc_domain: str,
    hashes: list[str],
    set_indexing: bool,
) -> None:
    # 1. Bulk insert → получаем chunk_ids
    chunk_ids = await uow.chunks.bulk_insert(
        document_id=document_id,
        filename=filename,
        visibility=visibility,
        chunks=[rc.page_content for rc in chunks],
        owner_id=owner_id,
        group_id=group_id,
        doc_domain=doc_domain,
        content_hashes=hashes,
    )

    # 2. Enrich metadata с chunk_ids
    for chunk_id, rc in zip(chunk_ids, chunks, strict=True):
        rc.metadata["chunk_id"] = chunk_id

    # 3. Enqueue outbox для Qdrant
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
    if set_indexing:
        await uow.documents.update_status(
            document_id,
            DocumentStatus.INDEXING.value,
            chunks=len(chunks),
            chars=total_chars,
        )
    else:
        # CLI: сразу done (outbox applying async, но CLI не ждёт)
        await uow.documents.update_status(
            document_id,
            DocumentStatus.DONE.value,
            chunks=len(chunks),
            chars=total_chars,
        )

    log.info(
        "Pipeline: doc %d — %d chunks enqueued (%d chars)",
        document_id,
        len(chunks),
        total_chars,
    )
