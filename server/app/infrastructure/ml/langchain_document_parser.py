"""LangChain document parser -- bridges RawDocument entities to LangChain Documents.

Wraps the low-level parsers from ``infrastructure.ml.ingestion`` and exposes a
``DocumentParser`` / ``DocumentSplitter`` pair compatible with the domain
service layer (``DocumentProcessor``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from domain.domain_profile.protocol import refs_to_metadata
from domain.entities.raw_document import RawDocument
from langchain.schema import Document

from infrastructure.ml.ingestion import (
    PARSERS,
    parse_docx,
    parse_docx_sections,
    parse_markdown_sections,
    parse_pdf,
)
from infrastructure.ml.ingestion import split_documents as _split_documents
from infrastructure.ml.rtf_decree_parser import parse_decree_rtf

if TYPE_CHECKING:
    from application.ports.domain_settings import DomainSettingsPort
    from domain.domain_profile.profiles.decree import DecreeDomainProfile
    from infrastructure.domain_profile.registry import DomainProfileRegistry

log = logging.getLogger("detailed")

_DECREE_FINGERPRINT_PREFIX_CHARS = 3000


def _lc_to_raw(docs) -> list[RawDocument]:
    return [RawDocument(page_content=d.page_content, metadata=dict(d.metadata)) for d in docs]


def _raw_to_lc(docs: list[RawDocument]):
    return [Document(page_content=d.page_content, metadata=d.metadata) for d in docs]


class LangchainDocumentParser:
    """Parses files into domain RawDocuments using LangChain infrastructure."""

    def __init__(
        self,
        domain_registry: "DomainProfileRegistry | None" = None,
        domain_settings: "DomainSettingsPort | None" = None,
    ) -> None:
        self._domain_registry = domain_registry
        self._domain_settings = domain_settings

    def parse(self, file_path: Path) -> list[RawDocument]:
        ext = file_path.suffix.lower()

        if ext == ".pdf":
            return _lc_to_raw(parse_pdf(file_path))

        if ext == ".md":
            return self._sections_to_documents(parse_markdown_sections(file_path), file_path)

        if ext in (".docx", ".doc"):
            sections = parse_docx_sections(file_path)
            # Get page metadata from docx page break detection
            _text, page_meta = parse_docx(file_path)
            return self._sections_to_documents(sections, file_path, page_meta)

        parser = PARSERS.get(ext)
        if parser is None:
            raise RuntimeError(f"Unsupported format: {ext}")

        result = parser(file_path)
        # Parsers may return str or tuple[str, dict]
        if isinstance(result, tuple):
            text, _meta = result
        else:
            text = result
        if not text or len(text.strip()) < 20:
            raise RuntimeError("Too little text in document")

        if ext == ".rtf":
            decree_docs = self._try_parse_decree_rtf(file_path, text)
            if decree_docs is not None:
                return decree_docs

        return [RawDocument(page_content=text, metadata={"source": file_path.name})]

    def _get_decree_profile(self) -> "DecreeDomainProfile | None":
        if self._domain_registry is None or self._domain_settings is None:
            return None
        try:
            profile = self._domain_registry.get("decree")
        except KeyError:
            return None
        return profile if profile.content_boundaries() else None

    def _try_parse_decree_rtf(self, file_path: Path, text: str) -> "list[RawDocument] | None":
        """Decree-structured RTF path; None = fall through to flat RTF parsing.

        Uses the same deterministic fingerprint as document classification,
        applied to the file prefix for speed (see TZ section 5.2).
        """
        profile = self._get_decree_profile()
        if profile is None or not profile.structural_fingerprint(text[:_DECREE_FINGERPRINT_PREFIX_CHARS]):
            return None
        try:
            units, doc_metadata = parse_decree_rtf(file_path, profile, self._domain_settings)
        except Exception:
            log.exception("Decree RTF parsing failed for %s — falling back to flat parse", file_path.name)
            return None

        docs = []
        for unit in units:
            if not unit.content.strip():
                continue
            metadata: dict = {"source": file_path.name, "unit_kind": unit.unit_kind}
            metadata.update(doc_metadata)
            if unit.heading:
                metadata["section"] = unit.heading
            if unit.boundary_value:
                metadata[f"{unit.unit_kind}_number"] = unit.boundary_value
            unit_refs = profile.extract_references(unit.content)
            if unit_refs:
                metadata["domain_metadata"] = refs_to_metadata(unit_refs)
            docs.append(RawDocument(page_content=unit.content, metadata=metadata))
        if not docs or sum(len(d.page_content) for d in docs) < 20:
            raise RuntimeError("Too little text in document")
        log.info("Decree RTF detected: %d structural units from %s", len(docs), file_path.name)
        return docs

    @staticmethod
    def _sections_to_documents(
        sections: list[tuple[str | None, str]],
        file_path: Path,
        page_meta: dict | None = None,
    ) -> list[RawDocument]:
        docs = []
        for heading, content in sections:
            if not content.strip():
                continue

            metadata: dict = {"source": file_path.name}
            if heading:
                metadata["section"] = heading
            if page_meta:
                metadata.update(page_meta)

            # Handle table blocks tagged by parse_markdown_sections
            if content.startswith("\x00TABLE:"):
                metadata["content_type"] = "table"
                content = content[len("\x00TABLE:") :]

            if not content.strip():
                continue
            docs.append(RawDocument(page_content=content, metadata=metadata))

        if not docs:
            raise RuntimeError("Too little text in document")
        return docs


class LangchainDocumentSplitter:
    """Splits domain RawDocuments into chunks using LangChain text splitters."""

    def __init__(
        self,
        domain_registry: "DomainProfileRegistry | None" = None,
        domain_settings: "DomainSettingsPort | None" = None,
    ) -> None:
        self._domain_registry = domain_registry
        self._domain_settings = domain_settings

    def split(self, documents: list[RawDocument], domain: str = "general") -> list[RawDocument]:
        profile = self._get_profile(domain)
        if (
            profile is not None
            and profile.content_boundaries()
            and documents
            and all("unit_kind" in d.metadata for d in documents)
        ):
            # Structural units produced by a domain-aware parser (decree RTF)
            # are already final chunks — pass through without re-splitting.
            return list(documents)
        return _lc_to_raw(
            _split_documents(
                _raw_to_lc(documents),
                domain=domain,
                profile=profile,
                settings=self._domain_settings,
            )
        )

    def _get_profile(self, domain: str):
        if self._domain_registry is None:
            return None
        try:
            return self._domain_registry.get(domain)
        except KeyError:
            return None
