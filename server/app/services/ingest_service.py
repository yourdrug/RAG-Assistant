"""
services/ingest_service.py — Ingestion orchestration, file upload, registry management.
"""

import logging
import time
from datetime import datetime
from pathlib import Path

from config import settings
from domain.ingestion import PARSERS, parse_pdf, split_documents
from infrastructure.clients import get_embeddings
from infrastructure.qdrant_ops import ensure_collection, upload_to_qdrant
from infrastructure.registry import (
    file_hash,
    is_already_indexed,
    load_registry,
    save_registry,
)
from infrastructure.storage import FileItem, get_storage
from langchain.schema import Document
from qdrant_client import QdrantClient

log = logging.getLogger("default")


def _tag_internal_public(chunks: list[Document]) -> None:
    for c in chunks:
        c.metadata.update({"visibility": "internal_public", "owner_id": None, "group_id": None})


class IngestService:
    def run_full_ingestion(self, docs_dir: str, reset: bool = False, prefix: str | None = None) -> None:
        t_start = time.monotonic()
        data_dir = settings.data_dir
        log.info("=" * 55)
        log.info("RAG Ingestion  |  mode: %s", "RESET" if reset else "APPEND")
        log.info("backend  : %s", settings.file_backend)
        if settings.file_backend == "s3":
            log.info("prefix   : %s (bucket: %s)", prefix or "docs/", settings.s3_bucket)
        else:
            log.info("docs_dir : %s", docs_dir)
        log.info("model    : %s", settings.embed_model)
        log.info("qdrant   : %s  /  collection: %s", settings.qdrant_url, settings.collection_name)
        log.info("=" * 55)

        registry = load_registry(data_dir)
        if reset:
            registry = {}

        embeddings = get_embeddings()
        vector_size = len(embeddings.embed_query("test"))
        qdrant_client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        ensure_collection(qdrant_client, vector_size, reset=reset)

        docs, cached = self._load_documents(docs_dir, registry, force=reset, prefix=prefix)
        if not docs:
            if cached > 0:
                log.info("All files already in registry — nothing to index. Use --reset to re-index.")
            else:
                log.error("No documents loaded. Check folder and formats.")
            return

        chunks = split_documents(docs)
        _tag_internal_public(chunks)
        upload_to_qdrant(chunks, embeddings)

        source_chars: dict[str, int] = {}
        for doc in docs:
            src = doc.metadata["source"]
            source_chars[src] = source_chars.get(src, 0) + len(doc.page_content)
        for src, chars in source_chars.items():
            if src.startswith("s3://"):
                fname = Path(src).name
                file_info = None
                if settings.file_backend == "s3":
                    key = "/".join(src.split("/")[3:])
                    file_info = get_storage().get_file_info(key)
                if file_info:
                    h = f"{file_info.size_bytes}_{file_info.last_modified}"
                else:
                    h = "unknown"
            else:
                path = Path(src)
                fname = path.name
                h = file_hash(path)
            registry[fname] = {
                "hash": h,
                "source": src,
                "chunks": sum(1 for c in chunks if c.metadata.get("source") == src),
                "chars": chars,
                "indexed_at": datetime.now().isoformat(timespec="seconds"),
            }
        save_registry(data_dir, registry)

        total_elapsed = time.monotonic() - t_start
        log.info("=" * 55)
        log.info("DONE  |  %d chunks  |  %.1fs total", len(chunks), total_elapsed)
        log.info("Registry: %d files", len(registry))
        log.info("=" * 55)

    def run_single_file(self, file_path: str) -> None:
        t_start = time.monotonic()
        data_dir = settings.data_dir

        log.info("=" * 55)
        log.info("RAG Ingestion  |  mode: SINGLE FILE")
        log.info("file     : %s", file_path)
        log.info("backend  : %s", settings.file_backend)
        log.info("=" * 55)

        registry = load_registry(data_dir)
        storage = get_storage()

        if settings.file_backend == "s3":
            key = file_path
            file_info = storage.get_file_info(key)
            if file_info is None:
                log.error("File not found in S3: %s", key)
                return
            if file_info.extension.lower() not in PARSERS:
                log.error("Unsupported format: %s", file_info.extension)
                return

            if not is_already_indexed(file_info, registry):
                temp_path = storage.download_to_temp(key)
                try:
                    docs = self._parse_file(file_info, temp_path)
                finally:
                    temp_path.unlink(missing_ok=True)
                if not docs:
                    log.error("Failed to parse file.")
                    return

                total_chars = sum(len(d.page_content) for d in docs)
                log.info("OK  %s  —  %s chars, %d pages", file_info.filename, f"{total_chars:,}", len(docs))

                chunks = self._index_docs(docs)
                registry[file_info.filename] = {
                    "hash": f"{file_info.size_bytes}_{file_info.last_modified}",
                    "source": f"s3://{settings.s3_bucket}/{key}",
                    "chunks": len(chunks),
                    "chars": total_chars,
                    "indexed_at": datetime.now().isoformat(timespec="seconds"),
                }
                save_registry(data_dir, registry)
            else:
                log.warning("File '%s' already in registry.", file_info.filename)
        else:
            path = Path(file_path)
            if not path.exists():
                log.error("File not found: %s", file_path)
                return
            if path.suffix.lower() not in PARSERS:
                log.error("Unsupported format: %s", path.suffix)
                return

            if is_already_indexed(path, registry):
                log.warning(
                    "File '%s' already in registry with same hash. Use --force to re-index.",
                    path.name,
                )
                return

            log.info("PARSE   %s  (%.1f KB)", path.name, path.stat().st_size / 1024)
            docs = self._parse_file(path)
            if not docs:
                log.error("Failed to parse file.")
                return

            total_chars = sum(len(d.page_content) for d in docs)
            log.info("OK  %s  —  %s chars, %d pages", path.name, f"{total_chars:,}", len(docs))

            chunks = self._index_docs(docs)
            registry[path.name] = {
                "hash": file_hash(path),
                "source": str(path),
                "chunks": len(chunks),
                "chars": total_chars,
                "indexed_at": datetime.now().isoformat(timespec="seconds"),
            }
            save_registry(data_dir, registry)

        log.info("=" * 55)
        log.info("DONE  |  %d chunks  |  %.1fs", len(chunks), time.monotonic() - t_start)
        log.info("=" * 55)

    def _index_docs(self, docs: list[Document]) -> list[Document]:
        embeddings = get_embeddings()
        vector_size = len(embeddings.embed_query("test"))
        client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        ensure_collection(client, vector_size, reset=False)

        chunks = split_documents(docs)
        _tag_internal_public(chunks)
        upload_to_qdrant(chunks, embeddings)
        return chunks

    def upload_files(self, files, prefix: str = "docs/") -> list[str]:
        storage = get_storage()
        uploaded: list[str] = []
        for f in files:
            key = prefix + f.filename
            data = f.data
            storage.upload_file(key, data)
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
        """
        Разрешает path_str (относительный или абсолютный) и гарантирует, что результат
        физически лежит внутри base. Проверка идёт по разрешённому (resolve()) пути через
        Path.relative_to, а не по строковому префиксу — иначе "/code/project/data-evil"
        прошёл бы наивную startswith-проверку "/code/project/data".
        """
        candidate = Path(path_str)
        resolved = candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()
        try:
            resolved.relative_to(base)
        except ValueError:
            raise ValueError(f"path must be inside {base} (DATA_DIR)") from None
        return resolved

    def resolve_ingest_target(self, file_path: str) -> str:
        if settings.file_backend == "s3":
            if file_path.startswith("/") or ".." in Path(file_path).parts:
                raise ValueError("file_path must not contain '..' or be absolute (S3 key)")
            return file_path

        base = Path(settings.data_dir).resolve()
        resolved = self._resolve_within_data_dir(file_path, base)
        if not resolved.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        return str(resolved)

    def resolve_docs_dir(self, docs_dir: str) -> str:
        """
        Валидирует docs_dir для POST /ingest (полная папка, не один файл).

        Без этой проверки admin — или утёкший/украденный admin-JWT (единственная
        авторизация тут) — мог бы передать docs_dir="/etc" или docs_dir="/" и
        рекурсивно затащить в общую коллекцию Qdrant любые читаемые процессом сервера
        файлы хоста. Дальше они станут доступны через обычный /chat любому user: у
        internal_public контента, которым помечается всё из /ingest (см.
        _tag_internal_public), нет ACL на уровне отдельного документа.
        """
        if settings.file_backend == "s3":
            # В S3-режиме run_full_ingestion не читает docs_dir с диска вообще —
            # индексация идёт по prefix внутри бакета (см. _load_documents).
            return docs_dir

        base = Path(settings.data_dir).resolve()
        resolved = self._resolve_within_data_dir(docs_dir, base)
        return str(resolved)

    def _parse_file(self, source, temp_path: Path | None = None) -> list[Document] | None:
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
            text = parser(path)
            if not text or len(text.strip()) < 20:
                log.warning("  SKIP  too little text: %s", fname)
                return None
            return [
                Document(
                    page_content=text,
                    metadata=self._base_metadata(source, path),
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
    ) -> tuple[list[Document], int]:
        storage = get_storage()

        if settings.file_backend == "s3":
            s3_prefix = prefix or "docs/"
            items = storage.list_files(s3_prefix)
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

        if settings.file_backend == "s3":
            for i, file_item in enumerate(items, 1):
                tag = f"[{i:>3}/{len(items)}]"
                if not force and is_already_indexed(file_item, registry):
                    log.info("%s CACHED  %s", tag, file_item.filename)
                    skipped_cached += 1
                    continue

                size_kb = file_item.size_bytes / 1024
                log.info("%s PARSE   %s  (%.1f KB)", tag, file_item.filename, size_kb)
                t0 = time.monotonic()

                temp_path = storage.download_to_temp(file_item.key)
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
                else:
                    errors += 1
        else:
            for i, file_path in enumerate(local_files, 1):
                tag = f"[{i:>3}/{len(local_files)}]"
                if not force and is_already_indexed(file_path, registry):
                    log.info("%s CACHED  %s", tag, file_path.name)
                    skipped_cached += 1
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
                else:
                    errors += 1

        log.info("Parsing complete: %d loaded, %d errors, %d already in registry", ok, errors, skipped_cached)
        return documents, skipped_cached
