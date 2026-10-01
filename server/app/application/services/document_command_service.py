"""DocumentCommandService — upload, delete, rename (write operations)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from application.dto.document_dto import DocumentDTO
from application.ports.bm25_index import BM25IndexPort
from application.ports.file_storage import FileStorage
from application.ports.unit_of_work_factory import UnitOfWorkFactory
from application.services.document_pipeline import build_outbox_metadata, enqueue_delete_by_document
from application.services.document_conflict_resolver import resolve_conflict_in_uow
from application.services.document_utils import generate_storage_key, resolve_unique_filename
from application.services.user_context_factory import UserContextFactory
from domain.entities.document import Document
from domain.entities.vector_outbox_entry import OutboxOperation, VectorOutboxEntry
from domain.exceptions import (
    BusinessRuleViolation,
    EntityNotFound,
    UniqueConstraintViolation,
    ValidationError,
)
from domain.repositories.vector_store_repository import VectorStoreRepository
from domain.services import check_ownership, compute_owner_and_group, validate_document_visibility
from domain.value_objects.doc_domain import DocDomain
from domain.value_objects.document_status import DocumentStatus
from domain.value_objects.roles import UserKind, UserRole
from domain.value_objects.visibility import DocumentVisibility

if TYPE_CHECKING:
    from domain.domain_profile.registry import DomainProfileRegistry

log = logging.getLogger(__name__)


class DocumentCommandService:
    """Write operations: upload, delete, rename."""

    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        vector_store_repo: VectorStoreRepository,
        file_storage: FileStorage,
        bm25_index: BM25IndexPort,
        domain_registry: "DomainProfileRegistry | None" = None,
        user_ctx_factory: UserContextFactory | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._vector_store = vector_store_repo
        self._file_storage = file_storage
        self._bm25_index = bm25_index
        self._domain_registry = domain_registry
        self._user_ctx_factory = user_ctx_factory or UserContextFactory()

    def _get_domain_profile(self, doc_domain: str):
        if self._domain_registry is None:
            return None
        try:
            return self._domain_registry.get(doc_domain)
        except KeyError:
            return None

    async def _remove_document_from_bm25(self, uow, document_id: int) -> None:
        chunks, _ = await uow.chunks.list_for_document(document_id, limit=10000)
        for chunk in chunks:
            if chunk.content_hash is not None:
                self._bm25_index.remove(chunk.content_hash)

    async def _resolve_effective_owner_id(
        self,
        uow,
        vis,
        user_id,
        user_kind,
        client_id,
        user_role=UserRole.USER,
        managed_client_ids=None,
    ) -> int:
        if vis != DocumentVisibility.CLIENT_PRIVATE:
            return user_id
        if user_kind == UserKind.CLIENT:
            return user_id
        if client_id is None:
            raise ValidationError("client_id required for client_private upload")
        client_user = await uow.users.get_by_id(client_id)
        if client_user is None or client_user.kind != UserKind.CLIENT:
            raise ValidationError("client_id must be a user with kind='client'")
        if user_role == UserRole.CURATOR:
            if managed_client_ids is None or client_id not in managed_client_ids:
                raise BusinessRuleViolation("You can only upload documents for assigned clients")
        return client_id

    @staticmethod
    def _validate_no_active_processing(existing) -> None:
        if existing and existing.status in (
            DocumentStatus.PENDING,
            DocumentStatus.PROCESSING,
            DocumentStatus.INDEXING,
        ):
            raise BusinessRuleViolation("This document is already being processed")

    async def _resolve_version_group(self, uow, existing, replaces_document_id, pending_replace_id):
        if replaces_document_id is not None:
            replaces_doc = await uow.documents.get_by_id(replaces_document_id)
            if replaces_doc is None:
                raise EntityNotFound("Document", replaces_document_id)
            version_group_id = replaces_doc.version_group_id or replaces_doc.id
            return version_group_id, pending_replace_id, replaces_doc
        if existing and existing.status in (DocumentStatus.DONE, DocumentStatus.FAILED):
            version_group_id = existing.version_group_id or existing.id
            return version_group_id, existing.id, existing
        return None, pending_replace_id, existing

    async def _persist_upload(self, uow, doc, file_data, owner_id, effective_group_id, filename):
        try:
            saved_doc = await uow.documents.save(doc)
        except UniqueConstraintViolation as exc:
            raise BusinessRuleViolation(
                "This document is already being uploaded by a concurrent request"
            ) from exc
        if saved_doc.id is None:
            raise RuntimeError("Document save returned None id")
        key = generate_storage_key(owner_id, effective_group_id, saved_doc.id, filename)
        try:
            await self._file_storage.upload_file(key, file_data)
            await uow.documents.set_source_path(saved_doc.id, key)
        except BaseException:
            try:
                await self._file_storage.delete_file(key)
            except Exception:
                log.warning("Failed to clean up orphaned upload object %s", key)
            raise
        return saved_doc, key

    async def _maybe_resolve_conflict_sync(
        self, doc, existing, pending_replace_id, doc_domain, uow, storage_deletes
    ):
        if doc_domain is None or existing is None or pending_replace_id is None:
            return pending_replace_id
        profile = self._get_domain_profile(doc_domain)
        old_source_path = await resolve_conflict_in_uow(
            uow,
            doc,
            existing,
            profile,
        )
        if old_source_path:
            storage_deletes.append(old_source_path)
        return None

    async def upload(
        self,
        filename,
        file_data,
        visibility,
        group_id,
        user_id,
        user_kind,
        user_role,
        client_id=None,
        rename_on_conflict=False,
        doc_domain=None,
        replaces_document_id=None,
    ) -> DocumentDTO:
        ext = Path(filename).suffix.lower()
        if ext not in self._file_storage.supported_extensions:
            raise ValidationError(f"Unsupported file format: {ext}")
        vis = DocumentVisibility.validate(visibility)
        storage_deletes: list[str] = []
        async with self._uow_factory.create(master=True) as uow:
            ctx = await self._user_ctx_factory.build(uow, user_id, user_kind, user_role)
            validate_document_visibility(vis, group_id, ctx)
            if vis == DocumentVisibility.INTERNAL_GROUP:
                if group_id is None:
                    raise ValidationError("group_id required for internal_group visibility")
                groups = await uow.groups.list_by_ids([group_id])
                if not groups:
                    raise EntityNotFound("Group", group_id)
            effective_owner_id = await self._resolve_effective_owner_id(
                uow,
                vis,
                user_id,
                user_kind,
                client_id,
                user_role=user_role,
                managed_client_ids=ctx.managed_client_ids if user_role == UserRole.CURATOR else None,
            )
            owner_id, effective_group_id = compute_owner_and_group(vis, group_id, effective_owner_id)
            existing = await uow.documents.find_active_slot(
                owner_id, filename, effective_group_id, for_update=True
            )
            self._validate_no_active_processing(existing)
            version_group_id, pending_replace_id, existing = await self._resolve_version_group(
                uow, existing, replaces_document_id, None
            )
            if existing and existing.status in (DocumentStatus.DONE, DocumentStatus.FAILED):
                filename = await resolve_unique_filename(
                    uow.documents, owner_id, effective_group_id, filename
                )
            doc = Document(
                filename=filename,
                visibility=vis,
                owner_id=owner_id,
                group_id=effective_group_id,
                doc_domain=doc_domain or DocDomain.GENERAL.value,
                version_group_id=version_group_id,
            )
            saved_doc, key = await self._persist_upload(
                uow, doc, file_data, owner_id, effective_group_id, filename
            )
            pending_replace_id = await self._maybe_resolve_conflict_sync(
                doc, existing, pending_replace_id, doc_domain, uow, storage_deletes
            )
            final_doc = await uow.documents.get_by_id(saved_doc.id)
            if final_doc is None:
                raise EntityNotFound("Document", saved_doc.id)
            dto = DocumentDTO.from_entity(final_doc, storage_key=key, replace_id=pending_replace_id)
        for old_key in storage_deletes:
            try:
                await self._file_storage.delete_file(old_key)
            except Exception:
                log.warning("Failed to delete replaced document object %s from storage — orphaned", old_key)
        return dto

    async def delete_document(
        self, document_id: int, user_id: int, user_role: str, user_kind: str = UserKind.INTERNAL
    ) -> None:
        async with self._uow_factory.create(master=True) as uow:
            doc = await uow.documents.get_by_id(document_id)
            if doc is None:
                raise EntityNotFound("Document", document_id)
            ctx = await self._user_ctx_factory.build(uow, user_id, user_kind, user_role)
            check_ownership(doc, ctx, "delete")
            await enqueue_delete_by_document(uow, document_id)
            await self._remove_document_from_bm25(uow, document_id)
            source_path = doc.source_path
            await uow.conversations.clear_summaries_referencing(document_id)
            await uow.documents.delete(document_id)
        if source_path:
            try:
                await self._file_storage.delete_file(source_path)
            except Exception:
                log.warning(
                    "Failed to delete storage object %s for document %d — orphaned", source_path, document_id
                )

    async def rename_document(
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
