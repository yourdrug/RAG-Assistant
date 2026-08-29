"""Application service for document lifecycle management.

Provides upload, rename, delete, permission management and storage-key
resolution for user documents. Each public method opens its own async
UnitOfWork via the injected UnitOfWorkFactory, keeping the service
stateless and transaction-safe.
"""

from __future__ import annotations

import logging
from pathlib import Path

from domain.entities.document import Document
from domain.exceptions import BusinessRuleViolation, EntityNotFound, ValidationError
from domain.repositories.vector_store_repository import VectorStoreRepository
from domain.services.access_control import (
    can_view_document,
    compute_owner_and_group,
    is_in_search_scope,
    validate_document_visibility,
)
from domain.value_objects.doc_domain import DocDomain
from domain.value_objects.document_status import DocumentStatus
from domain.value_objects.roles import UserKind, UserRole
from domain.value_objects.visibility import DocumentVisibility

from application.dto.document_dto import DocumentDTO
from application.ports.file_storage import FileStorage
from application.ports.unit_of_work_factory import UnitOfWorkFactory

log = logging.getLogger(__name__)


def to_document_dto(doc: Document, **overrides) -> DocumentDTO:
    """Convert a Document entity to a DocumentDTO."""
    assert doc.id is not None
    return DocumentDTO(
        id=doc.id,
        filename=doc.filename,
        visibility=doc.visibility,
        status=doc.status,
        source_path=doc.source_path,
        creation_date=doc.creation_date,
        indexed_at=doc.indexed_at,
        error_message=doc.error_message,
        warning_message=doc.warning_message,
        quality_score=doc.quality_score,
        chunks=doc.chunks,
        chars=doc.chars,
        owner_id=doc.owner_id,
        group_id=doc.group_id,
        doc_domain=doc.doc_domain,
        source_type=doc.source_type,
        has_manual_edits=doc.has_manual_edits,
        **overrides,
    )


async def check_document_access(uow, doc: Document, user_id: int, user_kind: str, user_role: str) -> None:
    """Raise BusinessRuleViolation if user cannot view the document."""
    user_group_ids = await uow.groups.get_user_group_ids(user_id) if user_kind == UserKind.INTERNAL else []
    if not can_view_document(
        doc_visibility=doc.visibility,
        doc_owner_id=doc.owner_id,
        doc_group_id=doc.group_id,
        user_kind=user_kind,
        user_id=user_id,
        user_group_ids=user_group_ids,
        user_role=user_role,
    ):
        raise BusinessRuleViolation("No access to this document")


class DocumentService:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        vector_store_repo: VectorStoreRepository,
        file_storage: FileStorage,
    ) -> None:
        self._uow_factory = uow_factory
        self._vector_store = vector_store_repo
        self._file_storage = file_storage

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

    async def _handle_existing_conflict(
        self,
        uow,
        existing,
        filename: str,
        owner_id: int | None,
        effective_group_id: int | None,
        rename_on_conflict: bool,
    ) -> tuple[str, int | None]:
        replace_id = None
        if not existing:
            return filename, replace_id
        if existing.status in (DocumentStatus.PENDING, DocumentStatus.PROCESSING):
            raise BusinessRuleViolation("This document is already being processed")
        if existing.status in (DocumentStatus.DONE, DocumentStatus.FAILED):
            if rename_on_conflict:
                filename = await self._unique_filename(uow, owner_id, effective_group_id, filename)
            else:
                replace_id = existing.id
                await self._vector_store.delete_by_document_id(existing.id)
                if existing.source_path:
                    self._file_storage.delete_file(existing.source_path)
                await uow.documents.delete(existing.id)
        return filename, replace_id

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
    ) -> DocumentDTO:
        vis = DocumentVisibility.validate(visibility)
        user_kind_enum = UserKind(user_kind)
        user_role_enum = UserRole(user_role)

        async with self._uow_factory.create(master=True) as uow:
            user_group_ids = await uow.groups.get_user_group_ids(user_id)
            validate_document_visibility(vis, group_id, user_kind_enum, user_role_enum, user_group_ids)

            if vis == DocumentVisibility.INTERNAL_GROUP:
                assert group_id is not None
                groups = await uow.groups.list_by_ids([group_id])
                if not groups:
                    raise EntityNotFound("Group", group_id)

            effective_owner_id = await self._resolve_effective_owner_id(
                uow,
                vis,
                user_id,
                user_kind,
                client_id,
            )

            owner_id, effective_group_id = compute_owner_and_group(vis, group_id, effective_owner_id)

            existing = await uow.documents.find_active_slot(
                owner_id, filename, effective_group_id, for_update=True
            )
            filename, replace_id = await self._handle_existing_conflict(
                uow,
                existing,
                filename,
                owner_id,
                effective_group_id,
                rename_on_conflict,
            )

            ext = Path(filename).suffix.lower()
            if ext not in self._file_storage.supported_extensions:
                raise ValidationError(f"Unsupported file format: {ext}")

            doc = Document(
                filename=filename,
                visibility=vis,
                owner_id=owner_id,
                group_id=effective_group_id,
                doc_domain=doc_domain or DocDomain.GENERAL.value,
            )

            try:
                saved_doc = await uow.documents.save(doc)
            except Exception as exc:
                if "unique" in str(exc).lower() or "integrity" in str(exc).lower():
                    raise BusinessRuleViolation(
                        "This document is already being uploaded by a concurrent request"
                    ) from exc
                raise

            assert saved_doc.id is not None
            key = self._storage_key(owner_id, effective_group_id, saved_doc.id, filename)
            await self._file_storage.upload_file(key, file_data)
            await uow.documents.set_source_path(saved_doc.id, key)

            final_doc = await uow.documents.get_by_id(saved_doc.id)
            assert final_doc is not None
            return to_document_dto(final_doc, storage_key=key, replace_id=replace_id)

    async def list_uploadable_clients(self, user_id: int, user_kind: str, user_role: str) -> list[dict]:
        async with self._uow_factory.create() as uow:
            if user_kind == UserKind.CLIENT:
                return []
            if user_role == UserRole.ADMIN:
                all_users = await uow.users.list_all()
                return [{"id": u.id, "email": u.email} for u in all_users if u.kind == UserKind.CLIENT]
            return []

    async def list_documents(
        self, user_id: int, user_kind: str, user_role: str | UserRole = UserRole.USER
    ) -> list[DocumentDTO]:
        async with self._uow_factory.create() as uow:
            if user_role == UserRole.ADMIN:
                docs = await uow.documents.list_all()
                admin_group_ids = await uow.groups.get_user_group_ids(user_id)
                return [
                    to_document_dto(
                        d,
                        in_search_scope=is_in_search_scope(
                            d.visibility,
                            d.owner_id,
                            d.group_id,
                            user_kind,
                            user_id,
                            admin_group_ids,
                            user_role,
                        ),
                    )
                    for d in docs
                ]
            elif user_kind == UserKind.CLIENT:
                docs = await uow.documents.list_visible(
                    user_kind=user_kind,
                    user_id=user_id,
                    group_ids=[],
                )
            else:
                group_ids = await uow.groups.get_user_group_ids(user_id)
                docs = await uow.documents.list_visible(
                    user_kind=user_kind,
                    user_id=user_id,
                    group_ids=group_ids or [],
                )

            return [to_document_dto(d) for d in docs]

    async def get_document(
        self, document_id: int, user_id: int, user_kind: str, user_role: str
    ) -> DocumentDTO:
        async with self._uow_factory.create() as uow:
            doc = await uow.documents.get_by_id(document_id)
            if doc is None:
                raise EntityNotFound("Document", document_id)

            await check_document_access(uow, doc, user_id, user_kind, user_role)

            return to_document_dto(doc)

    async def delete_document(self, document_id: int, user_id: int, user_role: str) -> None:
        async with self._uow_factory.create(master=True) as uow:
            doc = await uow.documents.get_by_id(document_id)
            if doc is None:
                raise EntityNotFound("Document", document_id)

            role = UserRole(user_role)
            user_group_ids = await uow.groups.get_user_group_ids(user_id)
            if not doc.can_be_deleted_by(user_id, role, user_group_ids):
                raise BusinessRuleViolation("Can only delete your own documents")

            await self._vector_store.delete_by_document_id(document_id)

            if doc.source_path:
                self._file_storage.delete_file(doc.source_path)

            await uow.documents.delete(document_id)

    async def list_source_files(self, search: str | None = None) -> list[str]:
        async with self._uow_factory.create() as uow:
            return await uow.documents.list_distinct_filenames(search=search, limit=100)

    @staticmethod
    async def _unique_filename(uow, owner_id, group_id, filename: str) -> str:
        p = Path(filename)
        stem = p.stem
        suffix = p.suffix
        candidate = filename
        counter = 1
        while await uow.documents.find_active_slot(owner_id, candidate, group_id) is not None:
            candidate = f"{stem}({counter}){suffix}"
            counter += 1
        if candidate != filename:
            log.info("Renamed conflict: %s -> %s", filename, candidate)
        return candidate

    @staticmethod
    def _storage_key(owner_id, group_id, document_id, filename):
        safe_name = Path(filename).name
        if owner_id is not None:
            return f"uploads/users/{owner_id}/{document_id}_{safe_name}"
        if group_id is not None:
            return f"uploads/groups/{group_id}/{document_id}_{safe_name}"
        return f"uploads/public/{document_id}_{safe_name}"
