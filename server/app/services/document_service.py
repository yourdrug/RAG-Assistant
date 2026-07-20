"""
services/document_service.py — Document upload, deletion, and background processing.
"""

import logging
from pathlib import Path

from config import settings
from domain.ingestion import PARSERS, parse_pdf, split_documents
from fastapi import HTTPException, UploadFile
from infrastructure.acl import can_view_document, owner_and_group_for, validate_visibility
from infrastructure.clients import get_embeddings
from infrastructure.database import (
    SessionLocal,
    create_document_row,
    delete_document_row,
    find_active_slot,
    get_document,
    list_documents_visible,
    set_document_source_path,
    update_document_status,
)
from infrastructure.qdrant_ops import ensure_collection, upload_to_qdrant
from infrastructure.storage import get_storage
from langchain.schema import Document
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from sqlalchemy.orm import Session

log = logging.getLogger("default")


class DocumentService:
    def _storage_key_for(
        self, owner_id: int | None, group_id: int | None, document_id: int, filename: str
    ) -> str:
        safe_name = Path(filename).name
        prefix = settings.uploads_prefix.rstrip("/")
        if owner_id is not None:
            return f"{prefix}/users/{owner_id}/{document_id}_{safe_name}"
        if group_id is not None:
            return f"{prefix}/groups/{group_id}/{document_id}_{safe_name}"
        return f"{prefix}/public/{document_id}_{safe_name}"

    async def save_upload(
        self, file: UploadFile, document_id: int, owner_id: int | None, group_id: int | None
    ) -> str:
        ext = Path(file.filename).suffix.lower()
        if ext not in settings.supported_extensions:
            raise HTTPException(400, f"Unsupported file format: {ext}")

        key = self._storage_key_for(owner_id, group_id, document_id, file.filename)
        data = await file.read()
        get_storage().upload_file(key, data)
        return key

    async def upload_document(
        self,
        file: UploadFile,
        visibility: str,
        group_id: int | None,
        user: dict,
        db: Session,
    ) -> dict:
        validate_visibility(visibility, group_id, user, db)
        owner_id, effective_group_id = owner_and_group_for(visibility, group_id, user)

        existing = find_active_slot(db, owner_id, file.filename, effective_group_id)
        if existing and existing["status"] in ("pending", "processing"):
            raise HTTPException(status_code=409, detail="This document is already being processed")
        replace_id = existing["id"] if existing and existing["status"] == "done" else None

        document_id = create_document_row(
            db,
            filename=file.filename,
            visibility=visibility,
            owner_id=owner_id,
            group_id=effective_group_id,
        )

        storage_key = await self.save_upload(file, document_id, owner_id, effective_group_id)
        set_document_source_path(db, document_id, storage_key)

        self._process_upload(
            document_id,
            storage_key,
            file.filename,
            visibility,
            owner_id,
            effective_group_id,
            replace_id,
        )

        return get_document(db, document_id)

    def list_documents(self, user: dict, db: Session) -> list[dict]:
        return list_documents_visible(db, user)

    def get_document(self, document_id: int, user: dict, db: Session) -> dict:
        doc = get_document(db, document_id)
        if doc is None:
            raise HTTPException(status_code=404, detail="Document not found")
        if not can_view_document(db, user, doc):
            raise HTTPException(status_code=403, detail="No access to this document")
        return doc

    def delete_document(self, document_id: int, user: dict, db: Session) -> None:
        doc = get_document(db, document_id)
        if doc is None:
            raise HTTPException(status_code=404, detail="Document not found")

        is_owner = doc["owner_id"] == user["id"]
        if not is_owner and user["role"] != "admin":
            raise HTTPException(status_code=403, detail="Can only delete your own documents")

        client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        client.delete(
            collection_name=settings.collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
            ),
        )
        if doc["source_path"]:
            get_storage().delete_file(doc["source_path"])
        delete_document_row(db, document_id)

    def process_upload(
        self,
        document_id: int,
        storage_key: str,
        original_filename: str,
        visibility: str,
        owner_id: int | None,
        group_id: int | None,
        replace_id: int | None,
    ) -> None:
        self._process_upload(
            document_id, storage_key, original_filename, visibility, owner_id, group_id, replace_id
        )

    def _process_upload(
        self,
        document_id: int,
        storage_key: str,
        original_filename: str,
        visibility: str,
        owner_id: int | None,
        group_id: int | None,
        replace_id: int | None,
    ) -> None:
        db = SessionLocal()
        temp_path: Path | None = None
        try:
            update_document_status(db, document_id, "processing")

            temp_path = get_storage().download_to_temp(storage_key)
            docs = self._parse_uploaded_file(temp_path, original_filename)
            if not docs:
                update_document_status(db, document_id, "failed", error="Could not extract text")
                return

            chunks = split_documents(docs)
            for chunk in chunks:
                chunk.metadata.update(
                    {
                        "document_id": document_id,
                        "visibility": visibility,
                        "owner_id": owner_id,
                        "group_id": group_id,
                    }
                )

            embeddings = get_embeddings()
            vector_size = len(embeddings.embed_query("test"))
            client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
            ensure_collection(client, vector_size, reset=False)

            upload_to_qdrant(chunks, embeddings)

            if replace_id is not None:
                client.delete(
                    collection_name=settings.collection_name,
                    points_selector=Filter(
                        must=[FieldCondition(key="document_id", match=MatchValue(value=replace_id))]
                    ),
                )
                old = get_document(db, replace_id)
                if old and old.get("source_path"):
                    get_storage().delete_file(old["source_path"])
                delete_document_row(db, replace_id)

            total_chars = sum(len(d.page_content) for d in docs)
            update_document_status(db, document_id, "done", chunks=len(chunks), chars=total_chars)

        except Exception as e:
            update_document_status(db, document_id, "failed", error=str(e))
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            db.close()

    def _parse_uploaded_file(self, local_path: Path, original_filename: str) -> list[Document]:
        ext = local_path.suffix.lower()

        if ext == ".pdf":
            docs = parse_pdf(local_path)
        else:
            parser = PARSERS.get(ext)
            if parser is None:
                raise RuntimeError(f"Unsupported format: {ext}")
            text = parser(local_path)
            if not text or len(text.strip()) < 20:
                raise RuntimeError("Too little text in document")
            docs = [Document(page_content=text, metadata={})]

        for doc in docs:
            doc.metadata["source"] = original_filename
        return docs
