"""Application service for document lifecycle management.

Provides upload, rename, delete, permission management and storage-key
resolution for user documents. Each public method opens its own async
UnitOfWork via the injected UnitOfWorkFactory, keeping the service
stateless and transaction-safe.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

from application.dto.document_dto import ClientInfo, DocumentDTO
from application.ports.bm25_index import BM25IndexPort
from application.ports.file_storage import FileStorage
from application.ports.unit_of_work_factory import UnitOfWorkFactory
from application.services.document_pipeline import build_outbox_metadata
from domain.entities.document import Document
from domain.entities.vector_outbox_entry import OutboxOperation, VectorOutboxEntry
from domain.exceptions import BusinessRuleViolation, EntityNotFound, ValidationError
from domain.repositories.vector_store_repository import VectorStoreRepository
from domain.services.access_control import (
    check_document_access,
    check_ownership,
    compute_owner_and_group,
    is_in_search_scope,
    validate_document_visibility,
)
from domain.services.document_utils import generate_storage_key, resolve_unique_filename
from domain.value_objects.doc_domain import DocDomain
from domain.value_objects.document_status import DocumentStatus
from domain.value_objects.roles import UserKind, UserRole
from domain.value_objects.user_context import UserContext
from domain.value_objects.visibility import DocumentVisibility

log = logging.getLogger(__name__)


class DocumentService:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        vector_store_repo: VectorStoreRepository,
        file_storage: FileStorage,
        bm25_index: BM25IndexPort,
        domain_registry=None,
        act_versioning_service=None,
    ) -> None:
        self._uow_factory = uow_factory
        self._vector_store = vector_store_repo
        self._file_storage = file_storage
        self._bm25_index = bm25_index
        self._domain_registry = domain_registry
        self._act_versioning_service = act_versioning_service

    def _get_domain_profile(self, doc_domain: str):
        """Get DomainProfile for the given domain, or None if not registered."""
        if self._domain_registry is None:
            return None
        try:
            return self._domain_registry.get(doc_domain)
        except KeyError:
            return None

    async def _build_user_context(self, uow, user_id: int, user_kind: str, user_role: str) -> UserContext:
        group_ids = await uow.groups.get_user_group_ids(user_id) if user_kind == UserKind.INTERNAL else []
        return UserContext(user_id=user_id, user_kind=user_kind, user_role=user_role, group_ids=group_ids)

    async def _remove_document_from_bm25(self, uow, document_id: int) -> None:
        """Remove all chunks of a document from the in-memory BM25 index."""
        chunks, _ = await uow.chunks.list_for_document(document_id, limit=10000)
        for chunk in chunks:
            if chunk.content_hash is not None:
                self._bm25_index.remove(chunk.content_hash)

    async def _enrich_with_outbox_status(self, uow, dto: DocumentDTO) -> DocumentDTO:
        """Enrich a DTO with outbox pending/failed counts for INDEXING documents."""
        if dto.status != DocumentStatus.INDEXING.value:
            return dto
        outbox = await uow.vector_outbox.count_by_document(dto.id)
        failed = None
        if outbox["failed"] > 0:
            failed = await uow.vector_outbox.get_failed_details(dto.id)
        return replace(
            dto,
            outbox_pending=outbox["pending"],
            outbox_failed=outbox["failed"],
            outbox_failed_details=failed,
        )

    async def _resolve_effective_owner_id(
            self,
            uow,
            vis: DocumentVisibility,
            user_id: int,
            user_kind: str,
            client_id: int | None,
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
        return client_id

    @staticmethod
    def _validate_no_active_processing(existing) -> None:
        """Reject if the slot is occupied by an in-flight document."""
        if existing and existing.status in (
            DocumentStatus.PENDING,
            DocumentStatus.PROCESSING,
            DocumentStatus.INDEXING,
        ):
            raise BusinessRuleViolation("This document is already being processed")

    async def _resolve_conflict_filename(
        self,
        uow,
        existing,
        owner_id: int | None,
        group_id: int | None,
        filename: str,
        rename_on_conflict: bool,
        versioned_domain: bool,
    ) -> str:
        """Return the (possibly unique-renamed) filename if a conflict exists."""
        if not existing or existing.status not in (DocumentStatus.DONE, DocumentStatus.FAILED):
            return filename
        if rename_on_conflict or versioned_domain:
            return await resolve_unique_filename(uow, owner_id, group_id, filename)
        return filename

    async def _replace_existing_document(
        self, uow, existing, storage_deletes: list[str] | None = None,
    ) -> int:
        """Delete the existing document's data in preparation for replace.

        BM25 cleanup → outbox enqueue → S3 deferral → DB delete.
        Returns the replaced document id.
        """
        await self._remove_document_from_bm25(uow, existing.id)
        await uow.vector_outbox.enqueue(
            VectorOutboxEntry(
                operation=OutboxOperation.DELETE_BY_DOCUMENT,
                aggregate_type="document",
                aggregate_id=existing.id,
                payload={"document_id": existing.id},
            )
        )
        if existing.source_path and storage_deletes is not None:
            storage_deletes.append(existing.source_path)
        await uow.documents.delete(existing.id)
        return existing.id

    async def upload(
        self,
        filename: str,
        file_data: bytes,
        visibility: str,
        group_id: int | None,
        user_id: int,
        user_kind: str,
        user_role: str,
        client_id: int | None = None,
        rename_on_conflict: bool = False,
        doc_domain: str | None = None,
        replaces_document_id: int | None = None,
    ) -> DocumentDTO:
        ext = Path(filename).suffix.lower()
        if ext not in self._file_storage.supported_extensions:
            raise ValidationError(f"Unsupported file format: {ext}")

        vis = DocumentVisibility.validate(visibility)

        storage_deletes: list[str] = []
        async with self._uow_factory.create(master=True) as uow:
            ctx = await self._build_user_context(uow, user_id, user_kind, user_role)
            validate_document_visibility(vis, group_id, ctx)

            if vis == DocumentVisibility.INTERNAL_GROUP:
                if group_id is None:
                    raise ValidationError("group_id required for internal_group visibility")
                groups = await uow.groups.list_by_ids([group_id])
                if not groups:
                    raise EntityNotFound("Group", group_id)

            effective_owner_id = await self._resolve_effective_owner_id(
                uow, vis, user_id, user_kind, client_id,
            )

            owner_id, effective_group_id = compute_owner_and_group(vis, group_id, effective_owner_id)

            existing = await uow.documents.find_active_slot(
                owner_id, filename, effective_group_id, for_update=True,
            )
            self._validate_no_active_processing(existing)

            # Determine version group and physical conflict resolution
            version_group_id: int | None = None
            pending_replace_id: int | None = None

            if replaces_document_id is not None:
                # Explicit signal: this is a new version of a specific document
                replaces_doc = await uow.documents.get_by_id(replaces_document_id)
                if replaces_doc is None:
                    raise EntityNotFound("Document", replaces_document_id)
                version_group_id = replaces_doc.version_group_id or replaces_doc.id
                existing = replaces_doc  # treat as the conflicting doc
            elif existing and existing.status in (DocumentStatus.DONE, DocumentStatus.FAILED):
                # Implicit signal: filename conflict → version group from existing
                version_group_id = existing.version_group_id or existing.id
                pending_replace_id = existing.id

            # Always rename on physical conflict (independent of domain knowledge)
            if existing and existing.status in (DocumentStatus.DONE, DocumentStatus.FAILED):
                filename = await resolve_unique_filename(uow, owner_id, effective_group_id, filename)

            doc = Document(
                filename=filename,
                visibility=vis,
                owner_id=owner_id,
                group_id=effective_group_id,
                doc_domain=doc_domain or DocDomain.GENERAL.value,
                version_group_id=version_group_id,
            )

            try:
                saved_doc = await uow.documents.save(doc)
            except Exception as exc:
                if "unique" in str(exc).lower() or "integrity" in str(exc).lower():
                    raise BusinessRuleViolation(
                        "This document is already being uploaded by a concurrent request"
                    ) from exc
                raise

            if saved_doc.id is None:
                raise RuntimeError("Document save returned None id")

            key = generate_storage_key(owner_id, effective_group_id, saved_doc.id, filename)
            await self._file_storage.upload_file(key, file_data)
            await uow.documents.set_source_path(saved_doc.id, key)

            # Sync resolve — if domain is already known
            if doc_domain is not None and existing and pending_replace_id is not None:
                from domain.services.document_versioning import resolve_conflict

                profile = self._get_domain_profile(doc_domain)
                await resolve_conflict(
                    self._uow_factory, doc, existing, profile,
                    act_versioning_service=self._act_versioning_service,
                )
                pending_replace_id = None  # resolved, no need for async

            final_doc = await uow.documents.get_by_id(saved_doc.id)
            if final_doc is None:
                raise EntityNotFound("Document", saved_doc.id)
            dto = DocumentDTO.from_entity(
                final_doc, storage_key=key, replace_id=pending_replace_id,
            )

        for old_key in storage_deletes:
            try:
                await self._file_storage.delete_file(old_key)
            except Exception:
                log.warning("Failed to delete replaced document object %s from storage — orphaned", old_key)
        return dto

    async def list_uploadable_clients(self, user_id: int, user_kind: str, user_role: str) -> list[ClientInfo]:
        async with self._uow_factory.create() as uow:
            if user_kind == UserKind.CLIENT:
                return []
            if user_role == UserRole.ADMIN:
                all_users = await uow.users.list_all()
                return [ClientInfo(id=u.id, email=u.email) for u in all_users if u.kind == UserKind.CLIENT]
            return []

    async def list_documents(
            self,
            user_id: int,
            user_kind: str,
            user_role: str | UserRole = UserRole.USER,
            limit: int = 200,
            offset: int = 0,
    ) -> list[DocumentDTO]:
        async with self._uow_factory.create() as uow:
            ctx = await self._build_user_context(uow, user_id, user_kind, user_role)
            if ctx.is_admin:
                docs = await uow.documents.list_all(limit=limit, offset=offset)
                dtos = [
                    DocumentDTO.from_entity(d, in_search_scope=is_in_search_scope(d, ctx))
                    for d in docs
                ]
            elif ctx.is_client:
                docs = await uow.documents.list_visible(
                    user_kind=user_kind, user_id=user_id, group_ids=[],
                    user_role=user_role, limit=limit, offset=offset,
                )
                dtos = [DocumentDTO.from_entity(d) for d in docs]
            else:
                docs = await uow.documents.list_visible(
                    user_kind=user_kind, user_id=user_id, group_ids=ctx.group_ids or [],
                    user_role=user_role, limit=limit, offset=offset,
                )
                dtos = [DocumentDTO.from_entity(d) for d in docs]

            return [await self._enrich_with_outbox_status(uow, dto) for dto in dtos]

    async def get_document(
        self, document_id: int, user_id: int, user_kind: str, user_role: str
    ) -> DocumentDTO:
        async with self._uow_factory.create() as uow:
            doc = await uow.documents.get_by_id(document_id)
            if doc is None:
                raise EntityNotFound("Document", document_id)

            ctx = await self._build_user_context(uow, user_id, user_kind, user_role)
            await check_document_access(uow, doc, ctx)

            dto = DocumentDTO.from_entity(doc)
            return await self._enrich_with_outbox_status(uow, dto)

    async def delete_document(self, document_id: int, user_id: int, user_role: str) -> None:
        async with self._uow_factory.create(master=True) as uow:
            doc = await uow.documents.get_by_id(document_id)
            if doc is None:
                raise EntityNotFound("Document", document_id)

            ctx = await self._build_user_context(uow, user_id, UserKind.INTERNAL, user_role)
            check_ownership(doc, ctx, "delete")

            # Enqueue vector store deletion via outbox (atomic with Postgres)
            await uow.vector_outbox.enqueue(
                VectorOutboxEntry(
                    operation=OutboxOperation.DELETE_BY_DOCUMENT,
                    aggregate_type="document",
                    aggregate_id=document_id,
                    payload={"document_id": document_id},
                )
            )

            # Remove chunks from BM25 index before DB cascade delete
            await self._remove_document_from_bm25(uow, document_id)

            source_path = doc.source_path

            # DB delete cascades to chunks via FK
            await uow.documents.delete(document_id)

        if source_path:
            try:
                await self._file_storage.delete_file(source_path)
            except Exception:
                log.warning(
                    "Failed to delete storage object %s for document %d — orphaned",
                    source_path,
                    document_id,
                )

    async def rename_document(
            self, document_id: int, new_filename: str, user_id: int, user_role: str
    ) -> DocumentDTO:
        ext = Path(new_filename).suffix.lower()
        if ext not in self._file_storage.supported_extensions:
            raise ValidationError(f"Unsupported file format: {ext}")

        async with self._uow_factory.create(master=True) as uow:
            doc = await uow.documents.get_by_id(document_id)
            if doc is None:
                raise EntityNotFound("Document", document_id)

            ctx = await self._build_user_context(uow, user_id, UserKind.INTERNAL, user_role)
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
                        uow, owner_id, effective_group_id, new_filename
                    )

            new_source_path = generate_storage_key(owner_id, effective_group_id, document_id, new_filename)

            old_source_path: str | None = None

            if doc.source_path and doc.source_path != new_source_path:
                await self._file_storage.copy_file(doc.source_path, new_source_path)
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
            dto = DocumentDTO.from_entity(final_doc)

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

    async def list_source_files(self, search: str | None = None) -> list[str]:
        async with self._uow_factory.create() as uow:
            return await uow.documents.list_distinct_filenames(search=search, limit=100)
