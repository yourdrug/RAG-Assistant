"""S3DocumentLoader -- file loading, parsing, splitting, and domain classification.

Extracted from IngestionService to isolate the S3 file I/O and parsing concern.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from application.ports.document_parser import DocumentParserPort, DocumentSplitterPort, FileMeta, SplitContext
from application.services.document_pipeline import classify_domain, tag_chunks, tag_domain
from application.services.ingestion_registry import s3_file_hash
from domain.entities.raw_document import RawDocument
from domain.services.document_domain_classifier import classify_document_domain
from domain.value_objects.doc_domain import DocDomain

if TYPE_CHECKING:
    from application.ports.domain_settings import DomainSettingsPort
    from application.ports.file_storage import FileItem, FileStorage
    from application.ports.ingestion_settings import IngestionSettingsPort
    from application.services.ingestion_registry import IngestionRegistry
    from domain.domain_profile.registry import DomainProfileRegistry

log = logging.getLogger("default")


class S3DocumentLoader:
    """Loads, parses, and splits documents from S3."""

    def __init__(
        self,
        file_storage: "FileStorage",
        parser: DocumentParserPort,
        splitter: DocumentSplitterPort,
        ingestion_settings: "IngestionSettingsPort",
        domain_registry: "DomainProfileRegistry | None" = None,
        domain_settings: "DomainSettingsPort | None" = None,
        registry: "IngestionRegistry | None" = None,
    ) -> None:
        self._file_storage = file_storage
        self._parser = parser
        self._splitter = splitter
        self._ingestion_settings = ingestion_settings
        self._domain_registry = domain_registry
        self._domain_settings = domain_settings
        self._registry = registry

    def s3_source_key(self, file_item: "FileItem") -> str:
        return f"s3://{self._ingestion_settings.s3_bucket}/{file_item.key}"

    def _parse_file(self, source: "FileItem", temp_path: Path) -> list | None:
        meta = FileMeta(
            source_key=self.s3_source_key(source),
            filename=source.filename,
            extension=source.extension,
            size_bytes=source.size_bytes,
        )
        try:
            docs = self._parser.parse(temp_path, meta)
            return docs if docs else None
        except Exception as e:
            log.error("  ERROR %s: %s", source.filename, e)
            return None

    def _base_metadata(self, source: "FileItem") -> dict:
        return {
            "source": self.s3_source_key(source),
            "filename": source.filename,
            "extension": source.extension,
            "size_bytes": source.size_bytes,
        }

    def split_docs(self, docs: list) -> list:
        """Split parsed documents into chunks via splitter port."""
        context = SplitContext(domain="general")
        return self._splitter.split(
            [RawDocument(page_content=d.page_content, metadata=dict(d.metadata)) for d in docs],
            context,
        )

    def index_docs(self, docs: list, domain: str = "general") -> list:
        """Parse and split documents into chunks. No direct Qdrant upload."""
        profile = self._get_profile(domain)
        context = SplitContext(
            domain=domain,
            profile=profile,
            settings=self._domain_settings,
            legal_mode=(domain == DocDomain.LEGAL.value),
        )
        chunks = self._splitter.split(
            [RawDocument(page_content=d.page_content, metadata=dict(d.metadata)) for d in docs],
            context,
        )
        tag_chunks(chunks)
        tag_domain(chunks, domain)
        return chunks

    def classify_text_domain(self, text: str, domain: str) -> str:
        """Registry-based classification with legacy fallback."""
        if domain != "auto":
            return domain
        return classify_domain(
            text,
            domain_registry=self._domain_registry,
            domain_settings=self._domain_settings,
            fallback_threshold=self._ingestion_settings.document_domain_marker_threshold,
            legacy_classifier=classify_document_domain,
        )

    def classify_source_domains(self, full_text_by_source: dict[str, str], domain: str) -> dict[str, str]:
        source_domain: dict[str, str] = {}
        for src, full_text in full_text_by_source.items():
            source_domain[src] = self.classify_text_domain(full_text, domain)
            log.info("doc_domain=%s for source=%s", source_domain[src], src)
        return source_domain

    def _get_profile(self, domain: str):
        if self._domain_registry is None:
            return None
        try:
            return self._domain_registry.get(domain)
        except KeyError:
            return None

    async def load_documents(
        self,
        registry: dict,
        force: bool = False,
        prefix: str | None = None,
    ) -> tuple[list, int]:
        """Load and parse all S3 files, returning (documents, skipped_count)."""
        s3_prefix = prefix or "docs/"
        items = self._file_storage.list_files(s3_prefix)
        log.info("Found %d files in s3://%s/%s", len(items), self._ingestion_settings.s3_bucket, s3_prefix)

        documents, skipped_cached, ok, errors = [], 0, 0, 0

        for i, file_item in enumerate(items, 1):
            tag = f"[{i:>3}/{len(items)}]"
            if (
                not force
                and self._registry is not None
                and await self._registry.is_indexed(file_item.filename, s3_file_hash(file_item))
            ):
                log.info("%s CACHED  %s", tag, file_item.filename)
                skipped_cached += 1
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
            else:
                errors += 1

        log.info("Parsing complete: %d loaded, %d errors, %d already in registry", ok, errors, skipped_cached)
        return documents, skipped_cached

    @staticmethod
    def validate_s3_key(key: str) -> None:
        if key.startswith("/") or ".." in Path(key).parts:
            raise ValueError("S3 key must not contain '..' or start with '/'")
