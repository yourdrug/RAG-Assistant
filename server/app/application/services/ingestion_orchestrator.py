"""S3 document ingestion pipeline.

Thin facade that orchestrates IngestionRegistry, DocumentSyncService,
and S3DocumentLoader.  Implements IngestionPort for CLI and API consumers.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from application.ports.document_parser import DocumentParserPort, DocumentSplitterPort
from application.ports.file_storage import FileStorage
from application.ports.ingestion_settings import IngestionSettingsPort
from application.ports.sparse_index import ChunkRef, SparseIndexAdminPort
from application.ports.unit_of_work_factory import UnitOfWorkFactory
from application.services.document_loader import S3DocumentLoader
from application.services.document_pipeline import classify_domain, tag_chunks, tag_domain  # noqa: F401 — re-export for test patching
from application.services.ingestion_registry import IngestionRegistry, s3_file_hash  # noqa: F401 — re-export
from application.services.ingestion_sync import DocumentSyncService
from domain.repositories.vector_store_repository import VectorStoreRepository
from domain.value_objects.doc_domain import DocDomain
from domain.value_objects.visibility import DocumentVisibility

log = logging.getLogger("default")


def _s3_file_hash(file_item) -> str:
    return f"{file_item.size_bytes}_{file_item.last_modified}"


class IngestionService:
    def __init__(
        self,
        vector_store_repo: VectorStoreRepository,
        file_storage: FileStorage,
        parser: DocumentParserPort,
        splitter: DocumentSplitterPort,
        ingestion_settings: IngestionSettingsPort,
        uow_factory: UnitOfWorkFactory | None = None,
        domain_registry: Any | None = None,
        domain_settings: Any | None = None,
        act_versioning_service: Any | None = None,
        sparse_index_admin: SparseIndexAdminPort | None = None,
    ) -> None:
        self._vector_store = vector_store_repo
        self._file_storage = file_storage
        self._parser = parser
        self._splitter = splitter
        self._ingestion_settings = ingestion_settings
        self._domain_registry = domain_registry
        self._domain_settings = domain_settings
        self._sparse_index_admin = sparse_index_admin

        # Collaborators
        if uow_factory is not None:
            self._registry = IngestionRegistry(uow_factory, file_storage)
            self._sync = DocumentSyncService(
                uow_factory, act_versioning_service, domain_registry, domain_settings
            )
        else:
            self._registry = None
            self._sync = None

        self._loader = S3DocumentLoader(
            file_storage,
            parser,
            splitter,
            ingestion_settings,
            domain_registry=domain_registry,
            domain_settings=domain_settings,
            registry=self._registry,
        )

    # -- Registry proxies (preserve existing callers) ----------------------

    async def _registry_list_all(self) -> dict:
        if self._registry is None:
            return {}
        return await self._registry.list_all()

    async def _registry_upsert(
        self,
        filename: str,
        file_hash_val: str,
        source: str,
        chunks_count: int,
        chars: int,
        indexed_at=None,
    ):
        if self._registry is None:
            return
        await self._registry.upsert(filename, file_hash_val, source, chunks_count, chars, indexed_at)

    async def _registry_is_indexed(self, filename: str, file_hash_val: str) -> bool:
        if self._registry is None:
            return False
        return await self._registry.is_indexed(filename, file_hash_val)

    async def _registry_delete(self, filename: str):
        if self._registry is None:
            return
        await self._registry.delete(filename)

    # -- Delegation --------------------------------------------------------

    def _s3_source_key(self, file_item) -> str:
        return self._loader.s3_source_key(file_item)

    def _log_ingest_config(self, reset: bool, docs_dir: str | None) -> None:
        log.info("=" * 55)
        log.info("RAG Ingestion  |  mode: %s", "RESET" if reset else "APPEND")
        log.info("backend  : s3")
        log.info("prefix   : %s (bucket: %s)", docs_dir or "docs/", self._ingestion_settings.s3_bucket)
        log.info("tei_embed: %s", self._ingestion_settings.tei_embed_url)
        qdrant = self._ingestion_settings.qdrant_url
        coll = self._ingestion_settings.collection_name
        log.info("qdrant   : %s  /  collection: %s", qdrant, coll)
        log.info("=" * 55)

    def _classify_text_domain(self, text: str, domain: str) -> str:
        return self._loader.classify_text_domain(text, domain)

    def _classify_source_domains(self, full_text_by_source: dict[str, str], domain: str) -> dict[str, str]:
        return self._loader.classify_source_domains(full_text_by_source, domain)

    def _collect_registry_entries(
        self,
        docs: list,
        chunks: list,
        source_chars: dict[str, int],
    ) -> dict[str, dict]:
        if self._registry is None:
            return {}
        return self._registry.collect_entries(docs, chunks, source_chars)

    async def _persist_registry_entries(self, entries: dict[str, dict]) -> None:
        if self._registry is None:
            return
        await self._registry.persist_entries(entries)

    async def _delete_internal_documents(self) -> None:
        if self._sync is None:
            return
        await self._sync.delete_internal_documents()

    async def _sync_documents_to_db(
        self,
        registry: dict,
        source_chars: dict,
        chunks: list,
        full_text_by_source: dict[str, str] | None = None,
        visibility: DocumentVisibility = DocumentVisibility.INTERNAL_PUBLIC,
        group_id: int | None = None,
        client_id: int | None = None,
    ) -> None:
        if self._sync is None:
            return
        await self._sync.sync_documents_to_db(
            registry,
            source_chars,
            chunks,
            full_text_by_source,
            visibility=visibility,
            group_id=group_id,
            client_id=client_id,
        )

    async def _split_docs(self, docs: list) -> list:
        return self._loader.split_docs(docs)

    async def _index_docs(self, docs: list, domain: str = "general") -> list:
        return self._loader.index_docs(docs, domain)

    def _get_profile(self, domain: str):
        if self._domain_registry is None:
            return None
        try:
            return self._domain_registry.get(domain)
        except KeyError:
            return None

    def _parse_file(self, source, temp_path: Path) -> list | None:
        return self._loader._parse_file(source, temp_path)

    def _base_metadata(self, source) -> dict:
        return self._loader._base_metadata(source)

    async def _load_documents(
        self,
        registry: dict,
        force: bool = False,
        prefix: str | None = None,
    ) -> tuple[list, int]:
        return await self._loader.load_documents(registry, force=force, prefix=prefix)

    # -- Public API (IngestionPort) ----------------------------------------

    async def run_full_ingestion(
        self,
        docs_dir: str | None = None,
        reset: bool = False,
        domain: str = "auto",
        visibility: DocumentVisibility = DocumentVisibility.INTERNAL_PUBLIC,
        group_id: int | None = None,
        client_id: int | None = None,
    ) -> None:
        t_start = time.monotonic()
        self._log_ingest_config(reset, docs_dir)

        registry = await self._registry_list_all()
        if reset:
            registry = {}

        await self._vector_store.ensure_collection(self._ingestion_settings.embed_dim, reset=reset)

        if reset:
            await self._delete_internal_documents()

        docs, cached = await self._load_documents(registry, force=reset, prefix=docs_dir)
        if not docs:
            if cached > 0:
                log.info("All files already in registry — nothing to index. Use --reset to re-index.")
            else:
                log.error("No documents loaded. Check S3 prefix and formats.")
            return

        chunks = await self._split_docs(docs)
        tag_chunks(chunks, visibility=visibility, owner_id=None, group_id=group_id, client_id=client_id)

        full_text_by_source: dict[str, str] = {}
        source_chars: dict[str, int] = {}
        for doc in docs:
            src = doc.metadata["source"]
            full_text_by_source[src] = full_text_by_source.get(src, "") + "\n" + doc.page_content
            source_chars[src] = source_chars.get(src, 0) + len(doc.page_content)

        source_domain = self._classify_source_domains(full_text_by_source, domain)

        for chunk in chunks:
            src = chunk.metadata.get("source", "")
            tag_domain([chunk], source_domain.get(src, DocDomain.GENERAL.value))

        entries = self._collect_registry_entries(docs, chunks, source_chars)
        registry = {**registry, **entries}

        await self._sync_documents_to_db(
            registry,
            source_chars,
            chunks,
            full_text_by_source,
            visibility=visibility,
            group_id=group_id,
            client_id=client_id,
        )
        await self._persist_registry_entries(entries)

        await self._build_sparse_index(chunks, reset=reset)

        total_elapsed = time.monotonic() - t_start
        log.info("=" * 55)
        log.info("DONE  |  %d chunks  |  %.1fs total", len(chunks), total_elapsed)
        log.info("=" * 55)

    async def _build_sparse_index(self, chunks: list, reset: bool = False) -> None:
        """Build and persist sparse index for hybrid search via port."""
        if not self._ingestion_settings.hybrid_enabled:
            return
        if self._sparse_index_admin is None:
            log.warning("SparseIndexAdminPort not injected — skipping sparse index build")
            return

        chunk_refs = [
            ChunkRef(
                text=c.page_content,
                content_hash=c.metadata.get("content_hash", ""),
                visibility=c.metadata.get("visibility", "internal_public"),
                owner_id=c.metadata.get("owner_id"),
                group_id=c.metadata.get("group_id"),
            )
            for c in chunks
        ]

        if reset:
            await self._sparse_index_admin.rebuild(chunk_refs)
        else:
            await self._sparse_index_admin.extend(chunk_refs)

    async def run_single_file(
        self,
        file_path: str,
        domain: str = "auto",
        visibility: DocumentVisibility = DocumentVisibility.INTERNAL_PUBLIC,
        group_id: int | None = None,
        client_id: int | None = None,
    ) -> None:
        t_start = time.monotonic()

        log.info("=" * 55)
        log.info("RAG Ingestion  |  mode: SINGLE FILE")
        log.info("file     : %s", file_path)
        log.info("backend  : s3")
        log.info("visibility: %s", visibility)
        log.info("=" * 55)

        registry = await self._registry_list_all()
        chunks = await self._handle_s3_file(
            file_path, domain, registry, visibility=visibility, group_id=group_id, client_id=client_id
        )

        if chunks is None:
            return

        log.info("=" * 55)
        log.info("DONE  |  %d chunks  |  %.1fs", len(chunks), time.monotonic() - t_start)
        log.info("=" * 55)

    async def _handle_s3_file(
        self,
        file_path: str,
        domain: str,
        registry: dict,
        visibility: DocumentVisibility = DocumentVisibility.INTERNAL_PUBLIC,
        group_id: int | None = None,
        client_id: int | None = None,
    ) -> list | None:
        key = file_path
        file_info = self._file_storage.get_file_info(key)
        if file_info is None:
            log.error("File not found in S3: %s", key)
            return None
        if file_info.extension.lower() not in self._file_storage.supported_extensions:
            log.error("Unsupported format: %s", file_info.extension)
            return None

        file_hash_str = _s3_file_hash(file_info)
        if await self._registry_is_indexed(file_info.filename, file_hash_str):
            log.warning("File '%s' already in registry.", file_info.filename)
            return None

        temp_path = await self._file_storage.download_to_temp(key)
        try:
            docs = self._parse_file(file_info, temp_path)
        finally:
            temp_path.unlink(missing_ok=True)
        if not docs:
            log.error("Failed to parse file.")
            return None

        total_chars = sum(len(d.page_content) for d in docs)
        log.info("OK  %s  —  %s chars, %d pages", file_info.filename, f"{total_chars:,}", len(docs))

        full_text = "\n".join(d.page_content for d in docs)
        file_domain = self._classify_text_domain(full_text, domain)
        log.info("doc_domain=%s for %s", file_domain, file_info.filename)

        chunks = await self._index_docs(docs, domain=file_domain)
        tag_chunks(chunks, visibility=visibility, owner_id=None, group_id=group_id, client_id=client_id)
        source = self._s3_source_key(file_info)

        entry = {
            file_info.filename: {
                "hash": file_hash_str,
                "source": source,
                "chunks": len(chunks),
                "chars": total_chars,
            }
        }
        await self._sync_documents_to_db(
            {**registry, **entry},
            {source: total_chars},
            chunks,
            {source: full_text},
            visibility=visibility,
            group_id=group_id,
            client_id=client_id,
        )
        await self._registry_upsert(
            file_info.filename,
            file_hash_str,
            source,
            len(chunks),
            total_chars,
        )
        return chunks

    async def upload_files(self, files, prefix: str = "docs/") -> list[str]:
        uploaded: list[str] = []
        for f in files:
            key = prefix + f.filename
            data = f.data
            await self._file_storage.upload_file(key, data)
            uploaded.append(key)
            log.info("Uploaded: %s (%d bytes)", key, len(data))
        return uploaded

    async def get_registry(self) -> dict:
        return await self._registry_list_all()

    async def force_reindex(self, filename: str) -> None:
        await self._registry_delete(filename)

    @staticmethod
    def _validate_s3_key(key: str) -> None:
        if key.startswith("/") or ".." in Path(key).parts:
            raise ValueError("S3 key must not contain '..' or start with '/'")

    def resolve_ingest_target(self, file_path: str) -> str:
        self._validate_s3_key(file_path)
        return file_path

    def resolve_docs_dir(self, docs_dir: str) -> str:
        return docs_dir
