"""Rename command: copy storage object and update document/chunks/outbox atomically."""

from __future__ import annotations

import logging
from pathlib import Path

from application.dto.document_dto import DocumentDTO
from application.ports.file_storage import FileStorage
from application.ports.unit_of_work_factory import UnitOfWorkFactory
from application.services.document_pipeline import build_outbox_metadata
from application.services.document_utils import generate_storage_key, resolve_unique_filename
from application.services.user_context_factory import UserContextFactory
from domain.entities.vector_outbox_entry import OutboxOperation, VectorOutboxEntry
from domain.exceptions import BusinessRuleViolation, EntityNotFound, ValidationError
from domain.services import check_ownership
from domain.value_objects.document_status import DocumentStatus
from domain.value_objects.roles import UserKind

log = logging.getLogger("default")


class DocumentRenameCommand:
    """Own rename transaction and compensation for a copied storage object."""

    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        file_storage: FileStorage,
        user_ctx_factory: UserContextFactory,
    ) -> None:
        self._uow_factory = uow_factory
        self._file_storage = file_storage
        self._user_ctx_factory = user_ctx_factory

    async def execute(
        self,
        document_id: int,
        new_filename: str,
        user_id: int,
        user_role: str,
        user_kind: str = UserKind.INTERNAL,
    ) -> DocumentDTO:
        ext = Path(new_filename).suffix.lower()
        if ext not in self._file_storage.supported_extensions:
            raise ValidationError(f"Unsupported file format: {ext}")
        copied_paths: list[str] = []
        try:
            async with self._uow_factory.create(master=True) as uow:
                dto, old_source_path = await self._rename_in_transaction(
                    uow, document_id, new_filename, user_id, user_role, user_kind, copied_paths
                )
        except BaseException:
            if copied_paths:
                try:
                    await self._file_storage.delete_file(copied_paths[0])
                except Exception:
                    log.warning(
                        "Failed to clean up orphaned copy %s for document %d", copied_paths[0], document_id
                    )
            raise
        if old_source_path:
            try:
                await self._file_storage.delete_file(old_source_path)
            except Exception:
                log.warning(
                    "Failed to delete old storage object %s after rename of document %d — orphaned",
                    old_source_path,
                    document_id,
                )
        return dto

    async def _rename_in_transaction(
        self, uow, document_id, new_filename, user_id, user_role, user_kind, copied_paths
    ):
        doc = await uow.documents.get_by_id(document_id)
        if doc is None:
            raise EntityNotFound("Document", document_id)
        ctx = await self._user_ctx_factory.build(uow, user_id, user_kind, user_role)
        check_ownership(doc, ctx, "rename")
        owner_id, effective_group_id = doc.owner_id, doc.group_id
        existing = await uow.documents.find_active_slot(
            owner_id, new_filename, effective_group_id, for_update=True
        )
        if existing and existing.id != document_id:
            if existing.status in (
                DocumentStatus.PENDING,
                DocumentStatus.PROCESSING,
                DocumentStatus.INDEXING,
            ):
                raise BusinessRuleViolation("This document name is already being processed")
            if existing.status in (DocumentStatus.DONE, DocumentStatus.FAILED):
                new_filename = await resolve_unique_filename(
                    uow.documents, owner_id, effective_group_id, new_filename
                )
        new_source_path = generate_storage_key(owner_id, effective_group_id, document_id, new_filename)
        old_source_path: str | None = None
        if doc.source_path and doc.source_path != new_source_path:
            await self._file_storage.copy_file(doc.source_path, new_source_path)
            copied_paths.append(new_source_path)
            old_source_path = doc.source_path
        await uow.documents.update_filename(document_id, new_filename, new_source_path)
        await uow.chunks.update_filename_by_document_id(document_id, new_filename)
        await uow.vector_outbox.enqueue(
            VectorOutboxEntry(
                operation=OutboxOperation.UPSERT_CHUNKS,
                aggregate_type="document",
                aggregate_id=document_id,
                payload={
                    "points": [
                        {
                            "chunk_id": c.chunk_id,
                            "page_content": c.content,
                            "metadata": build_outbox_metadata(
                                document_id=document_id,
                                visibility=c.visibility,
                                owner_id=c.owner_id,
                                group_id=c.group_id,
                                filename=new_filename,
                                doc_domain=c.doc_domain,
                                content_hash=c.content_hash,
                            ),
                        }
                        for c in (await uow.chunks.list_for_document(document_id, limit=10000))[0]
                    ]
                },
            )
        )
        final_doc = await uow.documents.get_by_id(document_id)
        if final_doc is None:
            raise EntityNotFound("Document", document_id)
        return DocumentDTO.from_entity(final_doc), old_source_path
