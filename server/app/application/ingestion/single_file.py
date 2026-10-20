"""Workflow for indexing one S3 document."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from application.indexing import tag_chunks
from application.ingestion.registry import s3_file_hash


if TYPE_CHECKING:
    from application.ingestion.dto import RegistryRecord
    from application.ingestion.loader import S3DocumentLoader
    from application.ingestion.registry import IngestionRegistry
    from application.ingestion.scope import IngestionScope
    from application.ingestion.sync import DocumentSyncService
    from application.ports.file_storage import FileStorage
    from domain.ingestion.raw_document import RawDocument

log = logging.getLogger("default")


class SingleFileIngestionWorkflow:
    """Coordinates the one-file parse, classification, and persistence path."""

    def __init__(
        self,
        file_storage: FileStorage,
        loader: S3DocumentLoader,
        registry: IngestionRegistry,
        sync: DocumentSyncService,
    ) -> None:
        self._file_storage = file_storage
        self._loader = loader
        self._registry = registry
        self._sync = sync

    async def run(self, file_path: str, *, scope: IngestionScope) -> None:
        started_at = time.monotonic()
        log.info("=" * 55)
        log.info("RAG Ingestion  |  mode: SINGLE FILE")
        log.info("file     : %s", file_path)
        log.info("backend  : s3")
        log.info("visibility: %s", scope.visibility)
        log.info("=" * 55)

        chunks = await self.ingest(file_path, scope=scope, registry=await self._registry.list_all())

        if chunks is None:
            return

        log.info("=" * 55)
        log.info("DONE  |  %d chunks  |  %.1fs", len(chunks), time.monotonic() - started_at)
        log.info("=" * 55)

    async def ingest(
        self, file_path: str, *, scope: IngestionScope, registry: dict[str, RegistryRecord]
    ) -> list[RawDocument] | None:
        file_info = await self._file_storage.get_file_info(file_path)

        if file_info is None:
            log.error("File not found in S3: %s", file_path)
        elif file_info.extension.lower() not in self._file_storage.supported_extensions:
            log.error("Unsupported format: %s", file_info.extension)

        if file_info is None or file_info.extension.lower() not in self._file_storage.supported_extensions:
            return None

        file_hash = s3_file_hash(file_info)

        if await self._registry.is_indexed(file_info.key, file_hash):
            log.warning("File '%s' already in registry.", file_info.filename)
            return None

        docs = await self._loader.load_file(file_info)

        if not docs:
            log.error("Failed to parse file.")
            return None

        total_chars = sum(len(doc.page_content) for doc in docs)
        log.info("OK  %s  —  %s chars, %d pages", file_info.filename, f"{total_chars:,}", len(docs))
        full_text = "\n".join(doc.page_content for doc in docs)
        domain = self._loader.classify_text_domain(full_text, scope.domain)
        log.info("doc_domain=%s for %s", domain, file_info.filename)

        chunks = self._loader.index_docs(docs, domain=domain)

        if not chunks:
            msg = f"No chunks prepared for {file_info.key}"
            raise ValueError(msg)

        tag_chunks(
            chunks,
            visibility=scope.visibility,
            owner_id=None,
            group_id=scope.group_id,
            client_id=scope.client_id,
        )
        source = self._loader.s3_source_key(file_info)
        entry: dict[str, RegistryRecord] = {
            file_info.key: {
                "filename": file_info.filename,
                "hash": file_hash,
                "source": source,
                "chunks": len(chunks),
                "chars": total_chars,
            }
        }
        await self._sync.sync_documents_to_db(
            {**registry, **entry},
            {source: total_chars},
            chunks,
            {source: full_text},
            visibility=scope.visibility,
            group_id=scope.group_id,
            client_id=scope.client_id,
        )

        # Record success only after sparse persistence, so failures remain retryable.
        await self._registry.upsert(file_info.key, file_hash, source, len(chunks), chars=total_chars)
        return chunks
