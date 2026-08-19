"""Infrastructure implementation of the full-document ingestion pipeline.

Scans a local directory or S3 bucket, parses each supported file, splits
into chunks, generates embeddings, uploads to Qdrant, builds a BM25 index
for hybrid search, and synchronises document metadata to Postgres via the
Unit-of-Work factory.  Supports both full-reset and incremental (append)
modes.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path

from config import settings
from domain.entities.chunk import Chunk
from domain.entities.document import Document as DocEntity
from domain.repositories.vector_store_repository import VectorStoreRepository
from domain.services.document_domain_classifier import classify_document_domain
from domain.value_objects.doc_domain import DocDomain
from domain.value_objects.file_backend import FileBackend
from domain.value_objects.visibility import DocumentVisibility
from langchain.schema import Document

from infrastructure.ml.hybrid import BM25Index, load_bm25_index, save_bm25_index
from infrastructure.ml.ingestion import (
    PARSERS,
    merge_pdf_pages,
    parse_pdf,
    split_documents,
    split_documents_legal,
)
from infrastructure.ml.metrics import INGEST_FILES_TOTAL
from infrastructure.registry import file_hash, is_already_indexed, load_registry, save_registry
from infrastructure.storage import FileItem, FileStorage
from infrastructure.uow_factory import UnitOfWorkFactory

log = logging.getLogger("default")


def _tag_internal_public(chunks: list) -> None:
    for c in chunks:
        c.metadata.update(
            {"visibility": DocumentVisibility.INTERNAL_PUBLIC.value, "owner_id": None, "group_id": None}
        )


def _tag_domain(chunks: list, doc_domain: str) -> None:
    for c in chunks:
        c.metadata["doc_domain"] = doc_domain


def _register_file(
    registry: dict, fname: str, file_hash: str, source: str, chunks_count: int, chars: int
) -> None:
    """Add or update a file entry in the ingestion registry."""
    registry[fname] = {
        "hash": file_hash,
        "source": source,
        "chunks": chunks_count,
        "chars": chars,
        "indexed_at": datetime.now().isoformat(timespec="seconds"),
    }


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

    @staticmethod
    def _log_ingest_config(docs_dir: str, reset: bool, prefix: str | None) -> None:
        log.info("=" * 55)
        log.info("RAG Ingestion  |  mode: %s", "RESET" if reset else "APPEND")
        log.info("backend  : %s", settings.file_backend)
        if settings.file_backend == FileBackend.S3.value:
            log.info("prefix   : %s (bucket: %s)", prefix or "docs/", settings.s3_bucket)
        else:
            log.info("docs_dir : %s", docs_dir)
        log.info("model    : %s", settings.embed_model)
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

    def _build_registry_entries(
        self,
        docs: list,
        chunks: list,
        source_chars: dict[str, int],
        registry: dict,
    ) -> None:
        for src, chars in source_chars.items():
            if src.startswith("s3://"):
                fname = Path(src).name
                file_info = None
                if settings.file_backend == FileBackend.S3.value:
                    key = "/".join(src.split("/")[3:])
                    file_info = self._file_storage.get_file_info(key)
                if file_info:
                    h = f"{file_info.size_bytes}_{file_info.last_modified}"
                else:
                    h = "unknown"
            else:
                path = Path(src)
                fname = path.name
                h = file_hash(path)
            chunks_count = sum(1 for c in chunks if c.metadata.get("source") == src)
            _register_file(registry, fname, h, src, chunks_count, chars)

    async def run_full_ingestion(
        self, docs_dir: str, reset: bool = False, prefix: str | None = None, domain: str = "auto"
    ) -> None:
        t_start = time.monotonic()
        data_dir = settings.data_dir
        self._log_ingest_config(docs_dir, reset, prefix)

        registry = load_registry(data_dir)
        if reset:
            registry = {}

        vector_size = len(await self._vector_store.generate_embeddings("test"))
        await self._vector_store.ensure_collection(vector_size, reset=reset)

        docs, cached = self._load_documents(docs_dir, registry, force=reset, prefix=prefix)
        if not docs:
            if cached > 0:
                log.info("All files already in registry — nothing to index. Use --reset to re-index.")
            else:
                log.error("No documents loaded. Check folder and formats.")
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
        self._build_registry_entries(docs, chunks, source_chars, registry)
        save_registry(data_dir, registry)
        await self._sync_documents_to_db(registry, source_chars, chunks)

        total_elapsed = time.monotonic() - t_start
        log.info("=" * 55)
        log.info("DONE  |  %d chunks  |  %.1fs total", len(chunks), total_elapsed)
        log.info("Registry: %d files", len(registry))
        log.info("=" * 55)

    async def _sync_documents_to_db(self, registry: dict, source_chars: dict, chunks: list) -> None:
        if self._uow_factory is None:
            return
        async with self._uow_factory.create(master=True) as uow:
            for fname, info in registry.items():
                src = info.get("source", "")
                existing = await uow.documents.find_active_slot(None, fname, None)
                if existing:
                    doc_id = existing.id
                    file_chunks = sum(1 for c in chunks if c.metadata.get("source") == src)
                    file_chars = source_chars.get(src, 0)
                    await uow.documents.update_status(
                        existing.id, "done", chunks=file_chunks, chars=file_chars
                    )
                else:
                    doc = DocEntity(filename=fname, visibility=DocumentVisibility.INTERNAL_PUBLIC)
                    saved = await uow.documents.save(doc)
                    doc_id = saved.id
                    file_chunks = info.get("chunks", 0)
                    file_chars = info.get("chars", 0)
                    await uow.documents.update_status(saved.id, "done", chunks=file_chunks, chars=file_chars)

                # Write chunks to Postgres for substring search
                file_chunk_texts = [c.page_content for c in chunks if c.metadata.get("source") == src]
                if file_chunk_texts and doc_id is not None:
                    first_chunk = next((c for c in chunks if c.metadata.get("source") == src), None)
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
                    await uow.chunks.bulk_insert(
                        document_id=doc_id,
                        filename=fname,
                        visibility=vis,
                        chunks=file_chunk_texts,
                        owner_id=owner,
                        group_id=group,
                        doc_domain=chunk_domain,
                    )
            log.info("Synced %d documents to database", len(registry))

    async def _upload_chunks_to_vector_store(self, chunks: list) -> None:
        """Convert LangChain Documents to domain Chunks and upload via repository.

        Also builds and persists the BM25 index for hybrid search.
        When adding to an existing index, merges new texts instead of overwriting.
        """
        domain_chunks = [Chunk(content=c.page_content, metadata=c.metadata) for c in chunks]
        await self._vector_store.upload_documents(domain_chunks)

        # Build and save BM25 index — merge with existing if present
        if settings.hybrid_enabled:
            bm25_path = Path(settings.data_dir) / "bm25_index.json"
            new_texts = [c.page_content for c in chunks]

            existing = load_bm25_index(bm25_path)
            if existing is not None:
                all_texts = existing.texts + new_texts
            else:
                all_texts = new_texts

            bm25_index = BM25Index(all_texts)
            save_bm25_index(bm25_index, bm25_path)
            # BM25 cache is now managed by MLClientRegistry — no manual clear needed

    @staticmethod
    def _classify_file_domain(docs: list, domain: str) -> str:
        if domain == "auto":
            full_text = "\n".join(d.page_content for d in docs)
            return classify_document_domain(full_text, threshold=settings.document_domain_marker_threshold)
        return domain

    async def _handle_s3_file(self, file_path: str, domain: str, registry: dict) -> list | None:
        key = file_path
        file_info = self._file_storage.get_file_info(key)
        if file_info is None:
            log.error("File not found in S3: %s", key)
            return None
        if file_info.extension.lower() not in settings.supported_extensions:
            log.error("Unsupported format: %s", file_info.extension)
            return None

        if is_already_indexed(file_info, registry):
            log.warning("File '%s' already in registry.", file_info.filename)
            return None

        temp_path = self._file_storage.download_to_temp(key)
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
        _register_file(
            registry,
            file_info.filename,
            f"{file_info.size_bytes}_{file_info.last_modified}",
            f"s3://{settings.s3_bucket}/{key}",
            len(chunks),
            total_chars,
        )
        save_registry(settings.data_dir, registry)
        await self._sync_documents_to_db(registry, {f"s3://{settings.s3_bucket}/{key}": total_chars}, chunks)
        return chunks

    async def _handle_local_file(self, file_path: str, domain: str, registry: dict) -> list | None:
        path = Path(file_path)
        if not path.exists():
            log.error("File not found: %s", file_path)
            return None
        if path.suffix.lower() not in settings.supported_extensions:
            log.error("Unsupported format: %s", path.suffix)
            return None

        if is_already_indexed(path, registry):
            log.warning(
                "File '%s' already in registry with same hash. Use --force to re-index.",
                path.name,
            )
            return None

        log.info("PARSE   %s  (%.1f KB)", path.name, path.stat().st_size / 1024)
        docs = self._parse_file(path)
        if not docs:
            log.error("Failed to parse file.")
            return None

        total_chars = sum(len(d.page_content) for d in docs)
        log.info("OK  %s  —  %s chars, %d pages", path.name, f"{total_chars:,}", len(docs))

        file_domain = self._classify_file_domain(docs, domain)
        log.info("doc_domain=%s for %s", file_domain, path.name)

        chunks = await self._index_docs(docs, domain=file_domain)
        _register_file(
            registry,
            path.name,
            file_hash(path),
            str(path),
            len(chunks),
            total_chars,
        )
        save_registry(settings.data_dir, registry)
        await self._sync_documents_to_db(registry, {str(path): total_chars}, chunks)
        return chunks

    async def run_single_file(self, file_path: str, domain: str = "auto") -> None:
        t_start = time.monotonic()

        log.info("=" * 55)
        log.info("RAG Ingestion  |  mode: SINGLE FILE")
        log.info("file     : %s", file_path)
        log.info("backend  : %s", settings.file_backend)
        log.info("=" * 55)

        registry = load_registry(settings.data_dir)

        if settings.file_backend == FileBackend.S3.value:
            chunks = await self._handle_s3_file(file_path, domain, registry)
        else:
            chunks = await self._handle_local_file(file_path, domain, registry)

        if chunks is None:
            return

        log.info("=" * 55)
        log.info("DONE  |  %d chunks  |  %.1fs", len(chunks), time.monotonic() - t_start)
        log.info("=" * 55)

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

    def upload_files(self, files, prefix: str = "docs/") -> list[str]:
        uploaded: list[str] = []
        for f in files:
            key = prefix + f.filename
            data = f.data
            self._file_storage.upload_file(key, data)
            uploaded.append(key)
            log.info("Uploaded: %s (%d bytes)", key, len(data))
        return uploaded

    def get_registry(self) -> dict:
        return load_registry(settings.data_dir)

    def force_reindex(self, filename: str) -> None:
        registry = load_registry(settings.data_dir)
        registry.pop(filename, None)
        save_registry(settings.data_dir, registry)

    @staticmethod
    def _resolve_within_data_dir(path_str: str, base: Path) -> Path:
        candidate = Path(path_str)
        resolved = candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()
        try:
            resolved.relative_to(base)
        except ValueError:
            raise ValueError(f"path must be inside {base} (DATA_DIR)") from None
        return resolved

    def resolve_ingest_target(self, file_path: str) -> str:
        if settings.file_backend == FileBackend.S3.value:
            if file_path.startswith("/") or ".." in Path(file_path).parts:
                raise ValueError("file_path must not contain '..' or be absolute (S3 key)")
            return file_path

        base = Path(settings.data_dir).resolve()
        resolved = self._resolve_within_data_dir(file_path, base)
        if not resolved.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        return str(resolved)

    def resolve_docs_dir(self, docs_dir: str) -> str:
        if settings.file_backend == FileBackend.S3.value:
            return docs_dir

        base = Path(settings.data_dir).resolve()
        resolved = self._resolve_within_data_dir(docs_dir, base)
        return str(resolved)

    def _parse_file(self, source, temp_path: Path | None = None):
        if isinstance(source, FileItem):
            path = temp_path or Path(source.filename)
            ext = source.extension
        else:
            path = source
            ext = path.suffix.lower()

        if ext == ".pdf":
            try:
                pages = parse_pdf(path)
                if not pages:
                    return None
                base = self._base_metadata(source, path)
                for doc in pages:
                    doc.metadata.update(base)
                return pages
            except Exception as e:
                log.error("  ERROR %s: %s", source.filename if isinstance(source, FileItem) else path.name, e)
                return None

        parser = PARSERS.get(ext)
        fname = source.filename if isinstance(source, FileItem) else path.name
        if parser is None:
            log.debug("  SKIP  unsupported format: %s", fname)
            return None
        try:
            result = parser(path)
            # Parsers may return str or tuple[str, dict]
            if isinstance(result, tuple):
                text, extra_meta = result
            else:
                text, extra_meta = result, {}
            if not text or len(text.strip()) < 20:
                log.warning("  SKIP  too little text: %s", fname)
                return None
            base = self._base_metadata(source, path)
            base.update(extra_meta)
            return [
                Document(
                    page_content=text,
                    metadata=base,
                )
            ]
        except Exception as e:
            log.error("  ERROR %s: %s", fname, e)
            return None

    def _base_metadata(self, source, temp_path: Path | None = None) -> dict:
        if isinstance(source, FileItem):
            return {
                "source": f"s3://{settings.s3_bucket}/{source.key}",
                "filename": source.filename,
                "extension": source.extension,
                "size_bytes": source.size_bytes,
            }
        return {
            "source": str(source),
            "filename": source.name,
            "extension": source.suffix.lower(),
            "size_bytes": source.stat().st_size,
        }

    def _load_documents(
        self,
        docs_dir: str,
        registry: dict,
        force: bool = False,
        prefix: str | None = None,
    ) -> tuple[list, int]:
        if settings.file_backend == FileBackend.S3.value:
            s3_prefix = prefix or "docs/"
            items = self._file_storage.list_files(s3_prefix)
            log.info("Found %d files in s3://%s/%s", len(items), settings.s3_bucket, s3_prefix)
        else:
            docs_path = Path(docs_dir)
            if not docs_path.exists():
                log.error("Folder not found: %s", docs_dir)
                return [], 0
            items = None
            local_files = sorted(
                f
                for f in docs_path.rglob("*")
                if f.is_file() and f.suffix.lower() in settings.supported_extensions
            )
            log.info("Found %d files in %s", len(local_files), docs_dir)

        documents, skipped_cached, ok, errors = [], 0, 0, 0

        if settings.file_backend == FileBackend.S3.value:
            for i, file_item in enumerate(items, 1):
                tag = f"[{i:>3}/{len(items)}]"
                if not force and is_already_indexed(file_item, registry):
                    log.info("%s CACHED  %s", tag, file_item.filename)
                    skipped_cached += 1
                    INGEST_FILES_TOTAL.labels(status="cached").inc()
                    continue

                size_kb = file_item.size_bytes / 1024
                log.info("%s PARSE   %s  (%.1f KB)", tag, file_item.filename, size_kb)
                t0 = time.monotonic()

                temp_path = self._file_storage.download_to_temp(file_item.key)
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
        else:
            for i, file_path in enumerate(local_files, 1):
                tag = f"[{i:>3}/{len(local_files)}]"
                if not force and is_already_indexed(file_path, registry):
                    log.info("%s CACHED  %s", tag, file_path.name)
                    skipped_cached += 1
                    INGEST_FILES_TOTAL.labels(status="cached").inc()
                    continue

                size_kb = file_path.stat().st_size / 1024
                log.info("%s PARSE   %s  (%.1f KB)", tag, file_path.name, size_kb)
                t0 = time.monotonic()

                docs = self._parse_file(file_path)
                elapsed = time.monotonic() - t0

                if docs:
                    documents.extend(docs)
                    total_chars = sum(len(d.page_content) for d in docs)
                    log.info(
                        "%s OK      %s — %s chars, %d pages, %.2fs",
                        tag,
                        file_path.name,
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
