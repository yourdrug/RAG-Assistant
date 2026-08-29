"""Infrastructure implementation of the S3-only document ingestion pipeline.

Scans an S3 bucket prefix, parses each supported file, splits into chunks,
generates embeddings, uploads to Qdrant, builds a BM25 index for hybrid
search, and synchronises document metadata to Postgres via the Unit-of-Work
factory.  Supports both full-reset and incremental (append) modes.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from config import settings
from domain.entities.chunk import Chunk
from domain.entities.document import Document as DocEntity
from domain.repositories.vector_store_repository import VectorStoreRepository
from domain.services.document_domain_classifier import classify_document_domain
from domain.value_objects.doc_domain import DocDomain
from domain.value_objects.visibility import DocumentVisibility
from langchain.schema import Document

from infrastructure.ml.hybrid import BM25Index, load_bm25_index_from_s3, save_bm25_index_to_s3
from infrastructure.ml.ingestion import (
    PARSERS,
    merge_pdf_pages,
    parse_pdf,
    split_documents,
    split_documents_legal,
)
from infrastructure.ml.metrics import INGEST_FILES_TOTAL
from infrastructure.storage import FileItem, FileStorage
from infrastructure.uow_factory import UnitOfWorkFactory

if TYPE_CHECKING:
    pass

log = logging.getLogger("default")


def _tag_internal_public(chunks: list) -> None:
    for c in chunks:
        c.metadata.update(
            {"visibility": DocumentVisibility.INTERNAL_PUBLIC.value, "owner_id": None, "group_id": None}
        )


def _tag_domain(chunks: list, doc_domain: str) -> None:
    for c in chunks:
        c.metadata["doc_domain"] = doc_domain


def _s3_file_hash(file_item: FileItem) -> str:
    return f"{file_item.size_bytes}_{file_item.last_modified}"


def _s3_source_key(file_item: FileItem) -> str:
    return f"s3://{settings.s3_bucket}/{file_item.key}"


class IngestionService:
    def __init__(
        self,
        vector_store_repo: VectorStoreRepository,
        file_storage: FileStorage,
        uow_factory: UnitOfWorkFactory | None = None,
    ) -> None:
        self._vector_store = vector_store_repo
        self._file_storage = file_storage
        self._uow_factory = uow_factory

    async def _registry_get(self, filename: str):
        if self._uow_factory is None:
            return None
        from infrastructure.repositories.sqlalchemy_ingestion_registry_repository import (
            SQLAlchemyIngestionRegistryRepository,
        )

        async with self._uow_factory.create(master=True) as uow:
            repo = SQLAlchemyIngestionRegistryRepository(uow._session)
            return await repo.get(filename)

    async def _registry_upsert(
        self, filename: str, file_hash_val: str, source: str, chunks_count: int, chars: int
    ):
        if self._uow_factory is None:
            return
        from datetime import datetime as _dt
        from domain.repositories.ingestion_registry_repository import IngestionRegistryEntry
        from infrastructure.repositories.sqlalchemy_ingestion_registry_repository import (
            SQLAlchemyIngestionRegistryRepository,
        )

        async with self._uow_factory.create(master=True) as uow:
            repo = SQLAlchemyIngestionRegistryRepository(uow._session)
            entry = IngestionRegistryEntry(
                filename=filename,
                file_hash=file_hash_val,
                source=source,
                chunks=chunks_count,
                chars=chars,
                indexed_at=_dt.now().isoformat(timespec="seconds"),
            )
            await repo.upsert(entry)

    async def _registry_is_indexed(self, filename: str, file_hash_val: str) -> bool:
        if self._uow_factory is None:
            return False
        from infrastructure.repositories.sqlalchemy_ingestion_registry_repository import (
            SQLAlchemyIngestionRegistryRepository,
        )

        async with self._uow_factory.create(master=True) as uow:
            repo = SQLAlchemyIngestionRegistryRepository(uow._session)
            return await repo.is_already_indexed(filename, file_hash_val)

    async def _registry_list_all(self) -> dict:
        if self._uow_factory is None:
            return {}
        from infrastructure.repositories.sqlalchemy_ingestion_registry_repository import (
            SQLAlchemyIngestionRegistryRepository,
        )

        async with self._uow_factory.create(master=True) as uow:
            repo = SQLAlchemyIngestionRegistryRepository(uow._session)
            entries = await repo.list_all()
            return {
                name: {
                    "hash": e.file_hash,
                    "source": e.source,
                    "chunks": e.chunks,
                    "chars": e.chars,
                    "indexed_at": e.indexed_at,
                }
                for name, e in entries.items()
            }

    async def _registry_delete(self, filename: str):
        if self._uow_factory is None:
            return
        from infrastructure.repositories.sqlalchemy_ingestion_registry_repository import (
            SQLAlchemyIngestionRegistryRepository,
        )

        async with self._uow_factory.create(master=True) as uow:
            repo = SQLAlchemyIngestionRegistryRepository(uow._session)
            await repo.delete(filename)

    @staticmethod
    def _log_ingest_config(reset: bool, docs_dir: str | None) -> None:
        log.info("=" * 55)
        log.info("RAG Ingestion  |  mode: %s", "RESET" if reset else "APPEND")
        log.info("backend  : s3")
        log.info("prefix   : %s (bucket: %s)", docs_dir or "docs/", settings.s3_bucket)
        log.info("tei_embed: %s", settings.tei_embed_url)
        log.info("qdrant   : %s  /  collection: %s", settings.qdrant_url, settings.collection_name)
        log.info("=" * 55)

    @staticmethod
    def _classify_source_domains(docs: list, domain: str) -> dict[str, str]:
        source_domain: dict[str, str] = {}
        for doc in docs:
            src = doc.metadata.get("source", "")
            if src not in source_domain:
                if domain == "auto":
                    source_domain[src] = classify_document_domain(
                        doc.page_content, threshold=settings.document_domain_marker_threshold
                    )
                else:
                    source_domain[src] = domain
                log.info("doc_domain=%s for source=%s", source_domain[src], src)
        return source_domain

    async def _build_registry_entries(
        self,
        docs: list,
        chunks: list,
        source_chars: dict[str, int],
    ) -> None:
        for src, chars in source_chars.items():
            fname = Path(src).name
            key = "/".join(src.split("/")[3:])
            file_info = self._file_storage.get_file_info(key)
            if file_info:
                h = _s3_file_hash(file_info)
            else:
                h = "unknown"
            chunks_count = sum(1 for c in chunks if c.metadata.get("source") == src)
            await self._registry_upsert(fname, h, src, chunks_count, chars)

    async def run_full_ingestion(
        self, docs_dir: str | None = None, reset: bool = False, domain: str = "auto"
    ) -> None:
        t_start = time.monotonic()
        self._log_ingest_config(reset, docs_dir)

        registry = await self._registry_list_all()
        if reset:
            registry = {}

        vector_size = len(await self._vector_store.generate_embeddings("test"))
        await self._vector_store.ensure_collection(vector_size, reset=reset)

        docs, cached = await self._load_documents(registry, force=reset, prefix=docs_dir)
        if not docs:
            if cached > 0:
                log.info("All files already in registry — nothing to index. Use --reset to re-index.")
            else:
                log.error("No documents loaded. Check S3 prefix and formats.")
            return

        chunks = split_documents(merge_pdf_pages(docs))
        _tag_internal_public(chunks)

        source_domain = self._classify_source_domains(docs, domain)

        for chunk in chunks:
            src = chunk.metadata.get("source", "")
            _tag_domain([chunk], source_domain.get(src, DocDomain.GENERAL.value))

        await self._upload_chunks_to_vector_store(chunks)

        source_chars: dict[str, int] = {}
        for doc in docs:
            src = doc.metadata["source"]
            source_chars[src] = source_chars.get(src, 0) + len(doc.page_content)
        await self._build_registry_entries(docs, chunks, source_chars)
        registry = await self._registry_list_all()
        await self._sync_documents_to_db(registry, source_chars, chunks)

        total_elapsed = time.monotonic() - t_start
        log.info("=" * 55)
        log.info("DONE  |  %d chunks  |  %.1fs total", len(chunks), total_elapsed)
        log.info("=" * 55)

    async def _sync_documents_to_db(self, registry: dict, source_chars: dict, chunks: list) -> None:
        if self._uow_factory is None:
            return
        async with self._uow_factory.create(master=True) as uow:
            for fname, info in registry.items():
                src = info.get("source", "")
                existing = await uow.documents.find_active_slot(None, fname, None)
                if existing:
                    assert existing.id is not None
                    doc_id = existing.id
                    file_chunks = sum(1 for c in chunks if c.metadata.get("source") == src)
                    file_chars = source_chars.get(src, 0)
                    await uow.documents.update_status(
                        existing.id, "done", chunks=file_chunks, chars=file_chars
                    )
                else:
                    doc = DocEntity(filename=fname, visibility=DocumentVisibility.INTERNAL_PUBLIC)
                    saved = await uow.documents.save(doc)
                    assert saved.id is not None
                    doc_id = saved.id
                    file_chunks = info.get("chunks", 0)
                    file_chars = info.get("chars", 0)
                    await uow.documents.update_status(saved.id, "done", chunks=file_chunks, chars=file_chars)

                file_chunks_list = [c for c in chunks if c.metadata.get("source") == src]
                file_chunk_texts = [c.page_content for c in file_chunks_list]
                if file_chunk_texts and doc_id is not None:
                    first_chunk = file_chunks_list[0] if file_chunks_list else None
                    vis = (
                        first_chunk.metadata.get("visibility", "internal_public")
                        if first_chunk
                        else "internal_public"
                    )
                    owner = first_chunk.metadata.get("owner_id") if first_chunk else None
                    group = first_chunk.metadata.get("group_id") if first_chunk else None
                    chunk_domain = (
                        first_chunk.metadata.get("doc_domain", DocDomain.GENERAL.value)
                        if first_chunk
                        else DocDomain.GENERAL.value
                    )
                    from domain.utils import content_hash as _content_hash

                    file_content_hashes = [_content_hash(t) for t in file_chunk_texts]
                    await uow.chunks.bulk_insert(
                        document_id=doc_id,
                        filename=fname,
                        visibility=vis,
                        chunks=file_chunk_texts,
                        owner_id=owner,
                        group_id=group,
                        doc_domain=chunk_domain,
                        content_hashes=file_content_hashes,
                    )
            log.info("Synced %d documents to database", len(registry))

    async def _upload_chunks_to_vector_store(self, chunks: list) -> None:
        """Convert LangChain Documents to domain Chunks and upload via repository.

        Also builds and persists the BM25 index for hybrid search.
        When adding to an existing index, merges new texts instead of overwriting.
        """
        domain_chunks = [Chunk(content=c.page_content, metadata=c.metadata) for c in chunks]
        await self._vector_store.upload_documents(domain_chunks)

        if settings.hybrid_enabled:
            new_texts = [c.page_content for c in chunks]

            existing = await load_bm25_index_from_s3(self._file_storage)
            if existing is not None:
                all_texts = existing.texts + new_texts
            else:
                all_texts = new_texts

            bm25_index = BM25Index(all_texts)
            await save_bm25_index_to_s3(bm25_index, self._file_storage)

    @staticmethod
    def _classify_file_domain(docs: list, domain: str) -> str:
        if domain == "auto":
            full_text = "\n".join(d.page_content for d in docs)
            return classify_document_domain(full_text, threshold=settings.document_domain_marker_threshold)
        return domain

    async def run_single_file(self, file_path: str, domain: str = "auto") -> None:
        t_start = time.monotonic()

        log.info("=" * 55)
        log.info("RAG Ingestion  |  mode: SINGLE FILE")
        log.info("file     : %s", file_path)
        log.info("backend  : s3")
        log.info("=" * 55)

        registry = await self._registry_list_all()
        chunks = await self._handle_s3_file(file_path, domain, registry)

        if chunks is None:
            return

        log.info("=" * 55)
        log.info("DONE  |  %d chunks  |  %.1fs", len(chunks), time.monotonic() - t_start)
        log.info("=" * 55)

    async def _handle_s3_file(self, file_path: str, domain: str, registry: dict) -> list | None:
        key = file_path
        file_info = self._file_storage.get_file_info(key)
        if file_info is None:
            log.error("File not found in S3: %s", key)
            return None
        if file_info.extension.lower() not in settings.supported_extensions:
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

        file_domain = self._classify_file_domain(docs, domain)
        log.info("doc_domain=%s for %s", file_domain, file_info.filename)

        chunks = await self._index_docs(docs, domain=file_domain)
        source = _s3_source_key(file_info)
        await self._registry_upsert(
            file_info.filename,
            file_hash_str,
            source,
            len(chunks),
            total_chars,
        )
        await self._sync_documents_to_db(registry, {source: total_chars}, chunks)
        return chunks

    async def _index_docs(self, docs: list, domain: str = "general") -> list:
        vector_size = len(await self._vector_store.generate_embeddings("test"))
        await self._vector_store.ensure_collection(vector_size, reset=False)

        merged = merge_pdf_pages(docs)
        if domain == DocDomain.LEGAL.value:
            chunks = split_documents_legal(merged)
        else:
            chunks = split_documents(merged)
        _tag_internal_public(chunks)
        _tag_domain(chunks, domain)
        await self._upload_chunks_to_vector_store(chunks)
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

    def _parse_file(self, source: FileItem, temp_path: Path) -> list | None:
        path = temp_path
        ext = source.extension

        if ext == ".pdf":
            try:
                pages = parse_pdf(path)
                if not pages:
                    return None
                base = self._base_metadata(source)
                for doc in pages:
                    doc.metadata.update(base)
                return pages
            except Exception as e:
                log.error("  ERROR %s: %s", source.filename, e)
                return None

        parser = PARSERS.get(ext)
        if parser is None:
            log.debug("  SKIP  unsupported format: %s", source.filename)
            return None
        try:
            result = parser(path)
            if isinstance(result, tuple):
                text, extra_meta = result
            else:
                text, extra_meta = result, {}
            if not text or len(text.strip()) < 20:
                log.warning("  SKIP  too little text: %s", source.filename)
                return None
            base = self._base_metadata(source)
            base.update(extra_meta)
            return [
                Document(
                    page_content=text,
                    metadata=base,
                )
            ]
        except Exception as e:
            log.error("  ERROR %s: %s", source.filename, e)
            return None

    def _base_metadata(self, source: FileItem) -> dict:
        return {
            "source": _s3_source_key(source),
            "filename": source.filename,
            "extension": source.extension,
            "size_bytes": source.size_bytes,
        }

    async def _load_documents(
        self,
        registry: dict,
        force: bool = False,
        prefix: str | None = None,
    ) -> tuple[list, int]:
        s3_prefix = prefix or "docs/"
        items = self._file_storage.list_files(s3_prefix)
        log.info("Found %d files in s3://%s/%s", len(items), settings.s3_bucket, s3_prefix)

        documents, skipped_cached, ok, errors = [], 0, 0, 0

        for i, file_item in enumerate(items, 1):
            tag = f"[{i:>3}/{len(items)}]"
            if not force and await self._registry_is_indexed(file_item.filename, _s3_file_hash(file_item)):
                log.info("%s CACHED  %s", tag, file_item.filename)
                skipped_cached += 1
                INGEST_FILES_TOTAL.labels(status="cached").inc()
                continue

            size_kb = file_item.size_bytes / 1024
            log.info("%s PARSE   %s  (%.1f KB)", tag, file_item.filename, size_kb)
            t0 = time.monotonic()

            temp_path = await self._file_storage.download_to_temp(file_item.key)
            try:
                docs = self._parse_file(file_item, temp_path)
            finally:
                temp_path.unlink(missing_ok=True)

            elapsed = time.monotonic() - t0
            if docs:
                documents.extend(docs)
                total_chars = sum(len(d.page_content) for d in docs)
                log.info(
                    "%s OK      %s — %s chars, %d pages, %.2fs",
                    tag,
                    file_item.filename,
                    f"{total_chars:,}",
                    len(docs),
                    elapsed,
                )
                ok += 1
                INGEST_FILES_TOTAL.labels(status="ok").inc()
            else:
                errors += 1
                INGEST_FILES_TOTAL.labels(status="error").inc()

        log.info("Parsing complete: %d loaded, %d errors, %d already in registry", ok, errors, skipped_cached)
        return documents, skipped_cached
