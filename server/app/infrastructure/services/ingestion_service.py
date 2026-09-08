"""Infrastructure implementation of the S3-only document ingestion pipeline.

Scans an S3 bucket prefix, parses each supported file, splits into chunks,
generates embeddings, uploads to Qdrant, builds a BM25 index for hybrid
search, and synchronises document metadata to Postgres via the Unit-of-Work
factory.  Supports both full-reset and incremental (append) modes.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from application.dto.versioning_dto import VersioningResult
from application.services.document_pipeline import classify_domain, enrich_chunks_metadata, process_chunks
from config import settings
from domain.entities.document import Document as DocEntity
from domain.entities.raw_document import RawDocument
from domain.entities.vector_outbox_entry import OutboxOperation, VectorOutboxEntry
from domain.repositories.ingestion_registry_repository import IngestionRegistryEntry
from domain.repositories.vector_store_repository import VectorStoreRepository
from domain.services.document_domain_classifier import classify_document_domain
from domain.value_objects.doc_domain import DocDomain
from domain.value_objects.visibility import DocumentVisibility
from langchain.schema import Document

from infrastructure.bm25.bm25_invalidation import publish_bm25_invalidation
from infrastructure.bm25.hybrid import BM25Index, load_bm25_index_from_s3, save_bm25_index_to_s3
from infrastructure.ml.ingestion import (
    PARSERS,
    merge_pdf_pages,
    parse_pdf,
    split_documents,
    split_documents_legal,
)
from infrastructure.metrics.metrics import INGEST_FILES_TOTAL
from infrastructure.repositories.sqlalchemy_ingestion_registry_repository import (
    SQLAlchemyIngestionRegistryRepository,
)
from infrastructure.storage import FileItem, FileStorage
from infrastructure.uow_factory import UnitOfWorkFactory

if TYPE_CHECKING:
    pass

log = logging.getLogger("default")


def _tag_chunks(
    chunks: list,
    visibility: DocumentVisibility = DocumentVisibility.INTERNAL_PUBLIC,
    owner_id: int | None = None,
    group_id: int | None = None,
    client_id: int | None = None,
) -> None:
    for c in chunks:
        c.metadata.update(
            {
                "visibility": visibility.value,
                "owner_id": owner_id,
                "group_id": group_id,
                "client_id": client_id,
            }
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
        domain_registry=None,
        domain_settings=None,
        act_versioning_service=None,
    ) -> None:
        self._vector_store = vector_store_repo
        self._file_storage = file_storage
        self._uow_factory = uow_factory
        self._domain_registry = domain_registry
        self._domain_settings = domain_settings
        self._act_versioning_service = act_versioning_service

    async def _registry_get(self, filename: str):
        if self._uow_factory is None:
            return None

        async with self._uow_factory.create(master=True) as uow:
            repo = SQLAlchemyIngestionRegistryRepository(uow._session)
            return await repo.get(filename)

    async def _registry_upsert(
        self,
        filename: str,
        file_hash_val: str,
        source: str,
        chunks_count: int,
        chars: int,
        indexed_at: datetime | None = None,
    ):
        if self._uow_factory is None:
            return

        async with self._uow_factory.create(master=True) as uow:
            repo = SQLAlchemyIngestionRegistryRepository(uow._session)
            entry = IngestionRegistryEntry(
                filename=filename,
                file_hash=file_hash_val,
                source=source,
                chunks=chunks_count,
                chars=chars,
                indexed_at=indexed_at or datetime.now(),
            )
            await repo.upsert(entry)

    async def _registry_is_indexed(self, filename: str, file_hash_val: str) -> bool:
        if self._uow_factory is None:
            return False

        async with self._uow_factory.create(master=True) as uow:
            repo = SQLAlchemyIngestionRegistryRepository(uow._session)
            return await repo.is_already_indexed(filename, file_hash_val)

    async def _registry_list_all(self) -> dict:
        if self._uow_factory is None:
            return {}

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

    def _classify_text_domain(self, text: str, domain: str) -> str:
        """Registry-based classification with legacy fallback.

        The SAME path the API upload uses (TZ section 2).
        """
        if domain != "auto":
            return domain

        return classify_domain(
            text,
            domain_registry=self._domain_registry,
            domain_settings=self._domain_settings,
            fallback_threshold=settings.document_domain_marker_threshold,
            legacy_classifier=classify_document_domain,
        )

    def _classify_source_domains(self, full_text_by_source: dict[str, str], domain: str) -> dict[str, str]:
        source_domain: dict[str, str] = {}
        for src, full_text in full_text_by_source.items():
            source_domain[src] = self._classify_text_domain(full_text, domain)
            log.info("doc_domain=%s for source=%s", source_domain[src], src)
        return source_domain

    def _collect_registry_entries(
        self,
        docs: list,
        chunks: list,
        source_chars: dict[str, int],
    ) -> dict[str, dict]:
        """Build registry entries in memory (no DB write).

        The registry row must only be committed AFTER the document/chunk
        sync transaction — a committed 'indexed' row without the data it
        claims exists makes the next incremental run report CACHED and
        silently skip the file forever.
        """
        entries: dict[str, dict] = {}
        for src, chars in source_chars.items():
            fname = Path(src).name
            key = "/".join(src.split("/")[3:])
            file_info = self._file_storage.get_file_info(key)
            if file_info:
                h = _s3_file_hash(file_info)
            else:
                h = "unknown"
            chunks_count = sum(1 for c in chunks if c.metadata.get("source") == src)
            entries[fname] = {
                "hash": h,
                "source": src,
                "chunks": chunks_count,
                "chars": chars,
            }
        return entries

    async def _persist_registry_entries(self, entries: dict[str, dict]) -> None:
        """Upsert collected registry entries (call after the sync committed)."""
        for fname, info in entries.items():
            await self._registry_upsert(
                fname,
                info["hash"],
                info["source"],
                info["chunks"],
                info["chars"],
                indexed_at=datetime.now(),
            )

    async def _delete_internal_documents(self) -> None:
        """Delete all internal documents (owner_id=NULL, non-manual) from DB.

        Called during reset to remove stale document records whose vectors
        have already been deleted from Qdrant. Manual documents are preserved.
        """
        if self._uow_factory is None:
            return
        async with self._uow_factory.create(master=True) as uow:
            deleted = await uow.documents.delete_internal_documents()
            log.info("Deleted %d internal documents from database", deleted)

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

        await self._vector_store.ensure_collection(settings.embed_dim, reset=reset)

        if reset:
            await self._delete_internal_documents()

        docs, cached = await self._load_documents(registry, force=reset, prefix=docs_dir)
        if not docs:
            if cached > 0:
                log.info("All files already in registry — nothing to index. Use --reset to re-index.")
            else:
                log.error("No documents loaded. Check S3 prefix and formats.")
            return

        chunks = split_documents(merge_pdf_pages(docs))
        _tag_chunks(chunks, visibility=visibility, owner_id=None, group_id=group_id, client_id=client_id)

        # Per-source full text — the same classification/versioning input the
        # API upload path uses (whole document, not a first chunk)
        full_text_by_source: dict[str, str] = {}
        source_chars: dict[str, int] = {}
        for doc in docs:
            src = doc.metadata["source"]
            full_text_by_source[src] = full_text_by_source.get(src, "") + "\n" + doc.page_content
            source_chars[src] = source_chars.get(src, 0) + len(doc.page_content)

        source_domain = self._classify_source_domains(full_text_by_source, domain)

        for chunk in chunks:
            src = chunk.metadata.get("source", "")
            _tag_domain([chunk], source_domain.get(src, DocDomain.GENERAL.value))

        # Collect registry entries in memory; persist them only AFTER the sync
        # transaction commits — otherwise a crash between upsert and sync
        # marks the file 'indexed' while nothing was ever stored, and the next
        # incremental run silently skips it as CACHED.
        entries = self._collect_registry_entries(docs, chunks, source_chars)
        registry = {**registry, **entries}

        # Sync to Postgres + enqueue outbox (Qdrant via dispatcher)
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

        # BM25 index (separate concern, not via outbox)
        await self._build_bm25_index(chunks, reset=reset)

        total_elapsed = time.monotonic() - t_start
        log.info("=" * 55)
        log.info("DONE  |  %d chunks  |  %.1fs total", len(chunks), total_elapsed)
        log.info("=" * 55)

    async def _build_bm25_index(self, chunks: list, reset: bool = False) -> None:
        """Build and persist BM25 index for hybrid search."""
        if not settings.hybrid_enabled:
            return

        new_texts = [c.page_content for c in chunks]

        if reset:
            all_texts = new_texts
            log.info("BM25: reset mode — rebuilding index from scratch (%d texts)", len(all_texts))
        else:
            existing = await load_bm25_index_from_s3(self._file_storage)
            if existing is not None:
                all_texts = existing.texts + new_texts
            else:
                all_texts = new_texts

        bm25_index = BM25Index(all_texts)
        await save_bm25_index_to_s3(bm25_index, self._file_storage)
        await publish_bm25_invalidation()

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
        """Sync documents and chunks to Postgres, enqueue outbox for Qdrant.

        Two phases so that act-version rows (created in their own transaction,
        exactly like the API upload path) can reference committed documents:
          Phase 1: resolve/create document rows (committed).
          Phase 2: per document — versioning (own tx) + chunks + outbox.
        """
        if self._uow_factory is None:
            return

        full_text_by_source = full_text_by_source or {}

        # Group chunks by source for per-document processing
        chunks_by_source: dict[str, list] = {}
        for c in chunks:
            src = c.metadata.get("source", "")
            chunks_by_source.setdefault(src, []).append(c)

        # --- Phase 1: document rows (single committed transaction) ---
        doc_ids = await self._ensure_document_rows(
            registry, visibility=visibility, group_id=group_id, client_id=client_id
        )

        # --- Phase 2: versioning + chunks + outbox (per document) ---
        for fname, info in registry.items():
            doc_id = doc_ids.get(fname)
            if doc_id is None:
                continue
            src = info.get("source", "")
            file_chunks = chunks_by_source.get(src, [])
            if not file_chunks:
                continue
            await self._sync_one_document(
                fname=fname,
                doc_id=doc_id,
                src=src,
                file_chunks=file_chunks,
                full_text=full_text_by_source.get(src, ""),
            )

        log.info("Synced %d documents to database via outbox", len(registry))

    async def _ensure_document_rows(
        self,
        registry: dict,
        visibility: DocumentVisibility = DocumentVisibility.INTERNAL_PUBLIC,
        group_id: int | None = None,
        client_id: int | None = None,
    ) -> dict[str, int]:
        """Phase 1: get-or-create document rows for all registry filenames."""
        doc_ids: dict[str, int] = {}
        assert self._uow_factory is not None
        async with self._uow_factory.create(master=True) as uow:
            # Determine owner_id based on visibility
            owner_id = None  # CLI ingest has no user context

            for fname in registry:
                # Use find_active_slot with correct owner_id/group_id to match unique constraint
                existing = await uow.documents.find_active_slot(
                    owner_id=owner_id,
                    filename=fname,
                    group_id=group_id,
                )
                if existing:
                    if existing.id is None:
                        raise RuntimeError(f"Document {fname} has None id after find_active_slot")
                    doc_ids[fname] = existing.id
                else:
                    doc = DocEntity(
                        filename=fname,
                        visibility=visibility,
                        owner_id=owner_id,
                        group_id=group_id,
                    )
                    saved = await uow.documents.save(doc)
                    if saved.id is None:
                        raise RuntimeError(f"Document {fname} has None id after save")
                    doc_ids[fname] = saved.id
        return doc_ids

    async def _sync_one_document(
        self,
        fname: str,
        doc_id: int,
        src: str,
        file_chunks: list,
        full_text: str,
    ) -> None:
        """Phase 2: versioning (own tx) + chunk persist + outbox, one document."""
        first_chunk = file_chunks[0]
        vis = first_chunk.metadata.get("visibility", "internal_public")
        owner = first_chunk.metadata.get("owner_id")
        group = first_chunk.metadata.get("group_id")
        chunk_domain = first_chunk.metadata.get("doc_domain", DocDomain.GENERAL.value)

        # Versioning — the same unified mechanism the API upload uses
        versioning = VersioningResult(
            domain_metadata=None,
            act_version_id=None,
            act_id=None,
            effective_from=None,
            warning=None,
        )
        profile = self._get_profile(chunk_domain)
        if self._act_versioning_service is not None and profile is not None:
            versioning = await self._act_versioning_service.process_document_versioning(
                profile, doc_id, full_text
            )
            if versioning.warning:
                log.warning("Versioning warning for %s: %s", fname, versioning.warning)

        raw_chunks = [
            RawDocument(page_content=c.page_content, metadata=dict(c.metadata)) for c in file_chunks
        ]

        assert self._uow_factory is not None
        async with self._uow_factory.create(master=True) as uow:
            enrich_chunks_metadata(
                raw_chunks,
                doc_id,
                vis,
                owner,
                group,
                chunk_domain,
                domain_metadata=versioning.domain_metadata,
                act_version_id=versioning.act_version_id,
                act_id=versioning.act_id,
                effective_from=versioning.effective_from,
            )

            # Save versioning warning to document
            if versioning.warning:
                await uow.documents.update_status(
                    doc_id,
                    status=None,
                    warning=versioning.warning,
                )

            await process_chunks(
                uow_factory=self._uow_factory,
                document_id=doc_id,
                filename=fname,
                chunks=raw_chunks,
                visibility=vis,
                owner_id=owner,
                group_id=group,
                doc_domain=chunk_domain,
                set_indexing=False,  # CLI: done immediately (outbox async)
                _existing_uow=uow,
            )

        # Link vectors to document ID — via the outbox: a vector-store
        # mutation must not run inline; a DB rollback must not leave Qdrant
        # patched with a document id that was never committed.
        await uow.vector_outbox.enqueue(
            VectorOutboxEntry(
                operation=OutboxOperation.SET_DOCUMENT_ID,
                aggregate_type="document",
                aggregate_id=doc_id,
                payload={"source": src, "document_id": doc_id},
            )
        )

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
        _tag_chunks(chunks, visibility=visibility, owner_id=None, group_id=group_id, client_id=client_id)
        source = _s3_source_key(file_info)

        # sync documents/chunks FIRST, persist the registry row only after
        # the sync transaction committed. A crash in between leaves no 'indexed'
        # row, so the next run re-indexes the file (safe) instead of skipping it
        # as CACHED (silent data loss).
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

    async def _index_docs(self, docs: list, domain: str = "general") -> list:
        """Parse and split documents into chunks. No direct Qdrant upload."""
        merged = merge_pdf_pages(docs)
        profile = self._get_profile(domain)
        if profile is not None and profile.content_boundaries() and self._domain_settings is not None:
            # Structured domain → content-based splitting (TZ section 4.3)
            chunks = split_documents(merged, domain=domain, profile=profile, settings=self._domain_settings)
        elif domain == DocDomain.LEGAL.value:
            chunks = split_documents_legal(merged)
        else:
            chunks = split_documents(merged)
        _tag_chunks(chunks)
        _tag_domain(chunks, domain)
        return chunks

    def _get_profile(self, domain: str):
        if self._domain_registry is None:
            return None
        try:
            return self._domain_registry.get(domain)
        except KeyError:
            return None

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
