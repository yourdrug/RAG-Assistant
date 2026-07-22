"""Document Processor — infrastructure implementation for processing uploaded documents.

Uses the caller's DB session (no separate transaction).
"""

from __future__ import annotations

import logging
from pathlib import Path

from config import settings
from domain.repositories.document_repository import DocumentRepository
from langchain.schema import Document
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from sqlalchemy.orm import Session

from infrastructure.clients import get_embeddings
from infrastructure.ml.ingestion import PARSERS, parse_pdf, split_documents
from infrastructure.qdrant_ops import ensure_collection, upload_to_qdrant
from infrastructure.storage import get_storage

log = logging.getLogger("default")


class DocumentProcessor:
    def __init__(self, db: Session, document_repo: DocumentRepository) -> None:
        self._db = db
        self._document_repo = document_repo

    def process(
        self,
        document_id: int,
        storage_key: str,
        original_filename: str,
        visibility: str,
        owner_id: int | None,
        group_id: int | None,
        replace_id: int | None,
    ) -> None:
        temp_path: Path | None = None
        try:
            self._document_repo.update_status(document_id, "processing")
            self._db.flush()

            temp_path = get_storage().download_to_temp(storage_key)
            docs = self._parse_uploaded_file(temp_path, original_filename)
            if not docs:
                self._document_repo.update_status(document_id, "failed", error="Could not extract text")
                self._db.flush()
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
                old = self._document_repo.get_by_id(replace_id)
                if old and old.source_path:
                    get_storage().delete_file(old.source_path)
                self._document_repo.delete(replace_id)

            total_chars = sum(len(d.page_content) for d in docs)
            self._document_repo.update_status(document_id, "done", chunks=len(chunks), chars=total_chars)
            self._db.flush()

        except Exception as e:
            try:
                self._document_repo.update_status(document_id, "failed", error=str(e))
                self._db.flush()
            except Exception:
                log.exception("Failed to mark document as failed")
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

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
