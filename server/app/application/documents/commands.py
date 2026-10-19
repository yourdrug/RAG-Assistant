"""Stable facade for document upload, deletion and rename commands."""

from __future__ import annotations

from typing import TYPE_CHECKING, Unpack

from domain.value_objects.roles import UserKind


if TYPE_CHECKING:
    from application.documents.delete import DocumentDeleteCommand
    from application.documents.dto import DocumentDTO, UploadOptions
    from application.documents.rename import DocumentRenameCommand
    from application.documents.upload import DocumentUploadCommand


class DocumentCommandService:
    """Delegate each write use case to a command with its own transaction boundary."""

    def __init__(
        self,
        upload_command: DocumentUploadCommand,
        delete_command: DocumentDeleteCommand,
        rename_command: DocumentRenameCommand,
    ) -> None:
        self._upload = upload_command
        self._delete = delete_command
        self._rename = rename_command

    async def upload(
        self,
        filename: str,
        file_data: bytes,
        visibility: str,
        group_id: int | None,
        **options: Unpack[UploadOptions],
    ) -> DocumentDTO:
        return await self._upload.execute(
            filename=filename,
            file_data=file_data,
            visibility=visibility,
            group_id=group_id,
            **options,
        )

    async def delete_document(
        self, document_id: int, user_id: int, user_role: str, user_kind: str = UserKind.INTERNAL
    ) -> None:
        await self._delete.execute(document_id, user_id, user_role, user_kind)

    async def rename_document(
        self,
        document_id: int,
        new_filename: str,
        user_id: int,
        user_role: str,
        user_kind: str = UserKind.INTERNAL,
    ) -> DocumentDTO:
        return await self._rename.execute(document_id, new_filename, user_id, user_role, user_kind)
