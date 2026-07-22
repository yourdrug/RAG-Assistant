"""Use Case: UploadDocument — upload and process a document."""

from __future__ import annotations

from pathlib import Path

from domain.entities.document import Document
from domain.exceptions import BusinessRuleViolation, ValidationError
from domain.repositories.document_repository import DocumentRepository
from domain.repositories.group_repository import GroupRepository
from domain.services.access_control import compute_owner_and_group, validate_document_visibility
from domain.value_objects.roles import UserKind, UserRole
from domain.value_objects.visibility import DocumentVisibility

from application.dto.document_dto import DocumentDTO


class UploadDocument:
    def __init__(
        self,
        document_repo: DocumentRepository,
        group_repo: GroupRepository,
        document_processor,
        file_storage,
    ) -> None:
        self._document_repo = document_repo
        self._group_repo = group_repo
        self._document_processor = document_processor
        self._file_storage = file_storage

    async def execute(
        self,
        filename: str,
        file_data: bytes,
        visibility: str,
        group_id: int | None,
        user_id: int,
        user_kind: str,
        user_role: str,
    ) -> DocumentDTO:
        vis = DocumentVisibility.validate(visibility)
        user_group_ids = self._group_repo.get_user_group_ids(user_id)

        validate_document_visibility(
            vis,
            group_id,
            UserKind(user_kind),
            UserRole(user_role),
            user_group_ids,
        )

        owner_id, effective_group_id = compute_owner_and_group(vis, group_id, user_id)

        existing = self._document_repo.find_active_slot(owner_id, filename, effective_group_id)
        replace_id = None
        if existing:
            if existing.status in ("pending", "processing"):
                raise BusinessRuleViolation("This document is already being processed")
            if existing.status == "done":
                replace_id = existing.id

        doc = Document(
            filename=filename,
            visibility=vis,
            owner_id=owner_id,
            group_id=effective_group_id,
        )
        saved_doc = self._document_repo.save(doc)

        key = self._storage_key(owner_id, effective_group_id, saved_doc.id, filename)
        ext = Path(filename).suffix.lower()
        if ext not in self._file_storage.supported_extensions:
            raise ValidationError(f"Unsupported file format: {ext}")

        self._file_storage.upload_file(key, file_data)
        self._document_repo.set_source_path(saved_doc.id, key)

        self._document_processor.process(
            document_id=saved_doc.id,
            storage_key=key,
            original_filename=filename,
            visibility=visibility,
            owner_id=owner_id,
            group_id=effective_group_id,
            replace_id=replace_id,
        )

        final_doc = self._document_repo.get_by_id(saved_doc.id)
        return DocumentDTO(
            id=final_doc.id,
            filename=final_doc.filename,
            visibility=final_doc.visibility,
            status=final_doc.status,
            error_message=final_doc.error_message,
            chunks=final_doc.chunks,
            chars=final_doc.chars,
        )

    @staticmethod
    def _storage_key(owner_id, group_id, document_id, filename):
        safe_name = Path(filename).name
        if owner_id is not None:
            return f"uploads/users/{owner_id}/{document_id}_{safe_name}"
        if group_id is not None:
            return f"uploads/groups/{group_id}/{document_id}_{safe_name}"
        return f"uploads/public/{document_id}_{safe_name}"
