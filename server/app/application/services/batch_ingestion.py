"""Workflow for indexing all new documents below an S3 prefix."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from application.services.document_pipeline import tag_chunks, tag_domain
from application.services.ingestion_scope import IngestionScope
from domain.value_objects.doc_domain import DocDomain

if TYPE_CHECKING:
    from application.ports.ingestion_settings import IngestionSettingsPort
    from application.services.document_loader import S3DocumentLoader
    from application.services.ingestion_sync import DocumentSyncService
    from application.services.ingestion_registry import IngestionRegistry
    from application.services.sparse_index_builder import SparseIndexBuilder
    from domain.repositories.vector_store_repository import VectorStoreRepository

log = logging.getLogger("default")


class BatchIngestionWorkflow:
    """Coordinates loading, chunking, database sync, and sparse projection."""

    def __init__(
        self,
        vector_store: "VectorStoreRepository",
        settings: "IngestionSettingsPort",
        loader: "S3DocumentLoader",
        registry: "IngestionRegistry",
        sync: "DocumentSyncService",
        sparse_index: "SparseIndexBuilder",
    ) -> None:
        self._vector_store = vector_store
        self._settings = settings
        self._loader = loader
        self._registry = registry
        self._sync = sync
        self._sparse_index = sparse_index

    async def run(self, docs_dir: str | None, *, reset: bool, scope: IngestionScope) -> None:
        started_at = time.monotonic()
        self._log_config(reset, docs_dir)

        registry = await self._registry.list_all()
        if reset:
            registry = {}

        await self._vector_store.ensure_collection(self._settings.embed_dim, reset=reset)
        if reset:
            await self._sync.delete_internal_documents()

        docs, cached = await self._loader.load_documents(registry, force=reset, prefix=docs_dir)
        if not docs:
            if cached:
                log.info("All files already in registry — nothing to index. Use --reset to re-index.")
            else:
                log.error("No documents loaded. Check S3 prefix and formats.")
            return

        full_text_by_source, source_chars = self._source_content(docs)
        source_domains = self._loader.classify_source_domains(full_text_by_source, scope.domain)
        chunks = self._loader.split_docs(docs, source_domains=source_domains)
        tag_chunks(
            chunks,
            visibility=scope.visibility,
            owner_id=None,
            group_id=scope.group_id,
            client_id=scope.client_id,
        )
        for chunk in chunks:
            source = chunk.metadata.get("source", "")
            tag_domain([chunk], source_domains.get(source, DocDomain.GENERAL.value))

        entries = self._registry.collect_entries(docs, chunks, source_chars)
        await self._sync.sync_documents_to_db(
            {**registry, **entries},
            source_chars,
            chunks,
            full_text_by_source,
            visibility=scope.visibility,
            group_id=scope.group_id,
            client_id=scope.client_id,
        )
        # Persist only after the document/chunk transaction commits.
        await self._registry.persist_entries(entries)
        await self._sparse_index.update(chunks, reset=reset)

        log.info("=" * 55)
        log.info("DONE  |  %d chunks  |  %.1fs total", len(chunks), time.monotonic() - started_at)
        log.info("=" * 55)

    def _log_config(self, reset: bool, docs_dir: str | None) -> None:
        log.info("=" * 55)
        log.info("RAG Ingestion  |  mode: %s", "RESET" if reset else "APPEND")
        log.info("backend  : s3")
        log.info("prefix   : %s (bucket: %s)", docs_dir or "docs/", self._settings.s3_bucket)
        log.info("tei_embed: %s", self._settings.tei_embed_url)
        log.info(
            "qdrant   : %s  /  collection: %s",
            self._settings.qdrant_url,
            self._settings.collection_name,
        )
        log.info("=" * 55)

    @staticmethod
    def _source_content(docs: list) -> tuple[dict[str, str], dict[str, int]]:
        full_text_by_source: dict[str, str] = {}
        source_chars: dict[str, int] = {}
        for doc in docs:
            source = doc.metadata["source"]
            full_text_by_source[source] = full_text_by_source.get(source, "") + "\n" + doc.page_content
            source_chars[source] = source_chars.get(source, 0) + len(doc.page_content)
        return full_text_by_source, source_chars
