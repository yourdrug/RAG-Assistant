"""Application Service: DocumentService — manages documents via UoWFactory.

Each method opens its own async UnitOfWork. No db/session parameters.
"""

from __future__ import annotations

import logging

from domain.exceptions import BusinessRuleViolation, EntityNotFound
from domain.repositories.vector_store_repository import VectorStoreRepository
from domain.services.access_control import (
    can_view_document,
    compute_owner_and_group,
    validate_document_visibility,
)
from domain.value_objects.roles import UserKind, UserRole
from domain.value_objects.visibility import DocumentVisibility
from infrastructure.storage import FileStorage
from infrastructure.uow_factory import UnitOfWorkFactory

from application.dto.document_dto import DocumentDTO

log = logging.getLogger(__name__)


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
    ) -> DocumentDTO:
        from pathlib import Path

        from domain.entities.document import Document
        from domain.exceptions import ValidationError
        from sqlalchemy.exc import IntegrityError

        vis = DocumentVisibility.validate(visibility)
        user_kind_enum = UserKind(user_kind)
        user_role_enum = UserRole(user_role)

        async with self._uow_factory.create() as uow:
            user_group_ids = await uow.groups.get_user_group_ids(user_id)
            validate_document_visibility(vis, group_id, user_kind_enum, user_role_enum, user_group_ids)

            if vis == DocumentVisibility.INTERNAL_GROUP:
                groups = await uow.groups.list_by_ids([group_id])
                if not groups:
                    raise EntityNotFound("Group", group_id)

            if vis == DocumentVisibility.CLIENT_PRIVATE:
                if user_kind == UserKind.CLIENT:
                    effective_owner_id = user_id
                else:
                    if client_id is None:
                        raise ValidationError("client_id required for client_private upload")
                    client_user = await uow.users.get_by_id(client_id)
                    if client_user is None or client_user.kind != UserKind.CLIENT:
                        raise ValidationError("client_id must be a user with kind='client'")
                    effective_owner_id = client_id
            else:
                effective_owner_id = user_id

            owner_id, effective_group_id = compute_owner_and_group(vis, group_id, effective_owner_id)

            existing = await uow.documents.find_active_slot(
                owner_id, filename, effective_group_id, for_update=True
            )
            replace_id = None
            if existing:
                if existing.status in ("pending", "processing"):
                    raise BusinessRuleViolation("This document is already being processed")
                if existing.status == "done":
                    if rename_on_conflict:
                        filename = await self._unique_filename(uow, owner_id, effective_group_id, filename)
                    else:
                        replace_id = existing.id
                        self._vector_store.delete_by_document_id(existing.id)
                        if existing.source_path:
                            self._file_storage.delete_file(existing.source_path)
                        await uow.documents.delete(existing.id)

            ext = Path(filename).suffix.lower()
            if ext not in self._file_storage.supported_extensions:
                raise ValidationError(f"Unsupported file format: {ext}")

            doc = Document(
                filename=filename,
                visibility=vis,
                owner_id=owner_id,
                group_id=effective_group_id,
            )

            try:
                saved_doc = await uow.documents.save(doc)
            except IntegrityError as exc:
                raise BusinessRuleViolation(
                    "This document is already being uploaded by a concurrent request"
                ) from exc

            key = self._storage_key(owner_id, effective_group_id, saved_doc.id, filename)
            self._file_storage.upload_file(key, file_data)
            await uow.documents.set_source_path(saved_doc.id, key)

            final_doc = await uow.documents.get_by_id(saved_doc.id)
            return DocumentDTO(
                id=final_doc.id,
                filename=final_doc.filename,
                visibility=final_doc.visibility,
                status=final_doc.status,
                source_path=final_doc.source_path,
                creation_date=final_doc.creation_date,
                indexed_at=final_doc.indexed_at,
                error_message=final_doc.error_message,
                chunks=final_doc.chunks,
                chars=final_doc.chars,
                storage_key=key,
                replace_id=replace_id,
                owner_id=owner_id,
                group_id=effective_group_id,
            )

    async def list_uploadable_clients(self, user_id: int, user_kind: str, user_role: str) -> list[dict]:
        async with self._uow_factory.create() as uow:
            if user_kind == UserKind.CLIENT:
                return []
            if user_role == UserRole.ADMIN:
                all_users = await uow.users.list_all()
                return [{"id": u.id, "email": u.email} for u in all_users if u.kind == UserKind.CLIENT]
            assigned_ids = await uow.client_assignments.get_assigned_client_ids(user_id)
            if not assigned_ids:
                return []
            clients = []
            for cid in assigned_ids:
                u = await uow.users.get_by_id(cid)
                if u and u.is_active:
                    clients.append({"id": u.id, "email": u.email})
            return clients

    async def list_documents(
        self, user_id: int, user_kind: str, user_role: str | UserRole = UserRole.USER
    ) -> list[DocumentDTO]:
        async with self._uow_factory.create() as uow:
            if user_role == UserRole.ADMIN:
                docs = await uow.documents.list_all()
            elif user_kind == UserKind.CLIENT:
                docs = await uow.documents.list_visible(
                    user_kind=user_kind,
                    user_id=user_id,
                    group_ids=[],
                    assigned_client_ids=[],
                )
            else:
                group_ids = await uow.groups.get_user_group_ids(user_id)
                assigned_ids = await uow.client_assignments.get_assigned_client_ids(user_id)
                docs = await uow.documents.list_visible(
                    user_kind=user_kind,
                    user_id=user_id,
                    group_ids=group_ids or [],
                    assigned_client_ids=assigned_ids or [],
                )

            return [
                DocumentDTO(
                    id=d.id,
                    filename=d.filename,
                    visibility=d.visibility,
                    status=d.status,
                    source_path=d.source_path,
                    creation_date=d.creation_date,
                    indexed_at=d.indexed_at,
                    error_message=d.error_message,
                    chunks=d.chunks,
                    chars=d.chars,
                    owner_id=d.owner_id,
                )
                for d in docs
            ]

    async def get_document(
        self, document_id: int, user_id: int, user_kind: str, user_role: str
    ) -> DocumentDTO:
        async with self._uow_factory.create() as uow:
            doc = await uow.documents.get_by_id(document_id)
            if doc is None:
                raise EntityNotFound("Document", document_id)

            user_group_ids = (
                await uow.groups.get_user_group_ids(user_id) if user_kind == UserKind.INTERNAL else []
            )
            assigned_ids = (
                await uow.client_assignments.get_assigned_client_ids(user_id)
                if user_kind == UserKind.INTERNAL
                else []
            )

            if not can_view_document(
                doc_visibility=doc.visibility,
                doc_owner_id=doc.owner_id,
                doc_group_id=doc.group_id,
                user_kind=user_kind,
                user_id=user_id,
                user_group_ids=user_group_ids,
                assigned_client_ids=assigned_ids,
            ):
                raise BusinessRuleViolation("No access to this document")

            return DocumentDTO(
                id=doc.id,
                filename=doc.filename,
                visibility=doc.visibility,
                status=doc.status,
                source_path=doc.source_path,
                creation_date=doc.creation_date,
                indexed_at=doc.indexed_at,
                error_message=doc.error_message,
                chunks=doc.chunks,
                chars=doc.chars,
            )

    async def delete_document(self, document_id: int, user_id: int, user_role: str) -> None:
        async with self._uow_factory.create() as uow:
            doc = await uow.documents.get_by_id(document_id)
            if doc is None:
                raise EntityNotFound("Document", document_id)

            role = UserRole(user_role)
            user_group_ids = await uow.groups.get_user_group_ids(user_id)
            if not doc.can_be_deleted_by(user_id, role, user_group_ids):
                raise BusinessRuleViolation("Can only delete your own documents")

            self._vector_store.delete_by_document_id(document_id)

            if doc.source_path:
                self._file_storage.delete_file(doc.source_path)

            await uow.documents.delete(document_id)

    @staticmethod
    async def _unique_filename(uow, owner_id, group_id, filename: str) -> str:
        from pathlib import Path

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
        from pathlib import Path

        safe_name = Path(filename).name
        if owner_id is not None:
            return f"uploads/users/{owner_id}/{document_id}_{safe_name}"
        if group_id is not None:
            return f"uploads/groups/{group_id}/{document_id}_{safe_name}"
        return f"uploads/public/{document_id}_{safe_name}"
