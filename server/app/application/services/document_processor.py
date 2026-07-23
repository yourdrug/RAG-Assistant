"""Document Processor — application service for processing uploaded documents.

Uses UoWFactory to manage its own transaction. No db/session parameters.
"""

from __future__ import annotations

import logging
from pathlib import Path

from domain.repositories.vector_store_repository import VectorStoreRepository
from domain.services.document_parser import DocumentParser, DocumentSplitter
from infrastructure.storage import FileStorage
from infrastructure.uow_factory import UnitOfWorkFactory

log = logging.getLogger("default")


class DocumentProcessor:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        vector_store_repo: VectorStoreRepository,
        file_storage: FileStorage,
        document_parser: DocumentParser,
        document_splitter: DocumentSplitter,
    ) -> None:
        self._uow_factory = uow_factory
        self._vector_store = vector_store_repo
        self._file_storage = file_storage
        self._parser = document_parser
        self._splitter = document_splitter

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
            with self._uow_factory.create() as uow:
                uow.documents.update_status(document_id, "processing")

                temp_path = self._file_storage.download_to_temp(storage_key)
                docs = self._parser.parse(temp_path)
                for doc in docs:
                    doc.metadata["source"] = original_filename

                chunks = self._splitter.split(docs)
                for chunk in chunks:
                    chunk.metadata.update(
                        {
                            "document_id": document_id,
                            "visibility": visibility,
                            "owner_id": owner_id,
                            "group_id": group_id,
                        }
                    )

                from domain.entities.chunk import Chunk

                domain_chunks = [Chunk(content=c.page_content, metadata=c.metadata) for c in chunks]

                vector_size = len(self._vector_store.generate_embeddings("test"))
                self._vector_store.ensure_collection(vector_size, reset=False)
                self._vector_store.upload_documents(domain_chunks)

                if replace_id is not None:
                    self._vector_store.delete_by_document_id(replace_id)
                    old = uow.documents.get_by_id(replace_id)
                    if old and old.source_path:
                        self._file_storage.delete_file(old.source_path)
                    uow.documents.delete(replace_id)

                total_chars = sum(len(d.page_content) for d in docs)
                uow.documents.update_status(document_id, "done", chunks=len(chunks), chars=total_chars)

        except Exception as e:
            log.exception("Document processing failed for doc %d: %s", document_id, e)
            try:
                with self._uow_factory.create() as uow:
                    uow.documents.update_status(document_id, "failed", error=str(e))
            except Exception:
                log.exception("Failed to mark document as failed")
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
