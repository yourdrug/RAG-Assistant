"""Application Service: DocumentService — orchestrates document use cases."""

from __future__ import annotations

from application.dto.document_dto import DocumentDTO
from application.use_cases.document.delete_document import DeleteDocument
from application.use_cases.document.get_document import GetDocument
from application.use_cases.document.list_documents import ListDocuments
from application.use_cases.document.upload_document import UploadDocument


class DocumentService:
    def __init__(
        self,
        upload_document: UploadDocument,
        list_documents: ListDocuments,
        get_document: GetDocument,
        delete_document: DeleteDocument,
    ) -> None:
        self._upload_document = upload_document
        self._list_documents = list_documents
        self._get_document = get_document
        self._delete_document = delete_document

    async def upload(
        self,
        filename: str,
        file_data: bytes,
        visibility: str,
        group_id: int | None,
        user_id: int,
        user_kind: str,
        user_role: str,
    ) -> DocumentDTO:
        return await self._upload_document.execute(
            filename=filename,
            file_data=file_data,
            visibility=visibility,
            group_id=group_id,
            user_id=user_id,
            user_kind=user_kind,
            user_role=user_role,
        )

    def list_documents(self, user_id: int, user_kind: str) -> list[DocumentDTO]:
        return self._list_documents.execute(user_id, user_kind)

    def get_document(self, document_id: int, user_id: int, user_kind: str, user_role: str) -> DocumentDTO:
        return self._get_document.execute(document_id, user_id, user_kind, user_role)

    def delete_document(self, document_id: int, user_id: int, user_role: str) -> None:
        self._delete_document.execute(document_id, user_id, user_role)
