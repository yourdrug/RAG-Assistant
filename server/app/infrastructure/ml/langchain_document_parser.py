"""LangChain document parser -- bridges RawDocument entities to LangChain Documents.

Wraps the low-level parsers from ``infrastructure.ml.ingestion`` and exposes a
``DocumentParser`` / ``DocumentSplitter`` pair compatible with the domain
service layer (``DocumentProcessor``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from langchain.schema import Document

from domain.entities.raw_document import RawDocument
from domain.value_objects.doc_domain import DocDomain
from infrastructure.ml.ingestion import (
    PARSERS,
    parse_docx,
    parse_docx_sections,
    parse_markdown_sections,
    parse_pdf,
)
from infrastructure.ml.ingestion import split_documents as _split_documents
from infrastructure.ml.ingestion.amendment_context import attach_table_replacement_scope
from infrastructure.ml.ingestion.markdown import extract_doc_title as _extract_md_title
from infrastructure.ml.ingestion.rtf import extract_doc_title as _extract_rtf_title
from infrastructure.ml.ingestion.rtf import parse_rtf, parse_rtf_sections
from infrastructure.ml.ingestion.structural_metadata import unit_metadata
from infrastructure.ml.ingestion.markdown import extract_date_from_filename
from infrastructure.ml.rtf_decree_parser import parse_decree_rtf

if TYPE_CHECKING:
    from domain.domain_profile.settings_port import DomainSettingsPort
    from domain.domain_profile.profiles.decree import DecreeDomainProfile
    from domain.domain_profile.registry import DomainProfileRegistry

log = logging.getLogger("detailed")

_DECREE_FINGERPRINT_PREFIX_CHARS = 3000

# Keyword-based document-type classification — delegated to domain layer.
from domain.services.document_domain_classifier import (  # noqa: E402
    DOC_TYPE_SAMPLE_CHARS,
    classify_doc_type,
)


def _lc_to_raw(docs) -> list[RawDocument]:
    return [RawDocument(page_content=d.page_content, metadata=dict(d.metadata)) for d in docs]


def _raw_to_lc(docs: list[RawDocument]):
    return [Document(page_content=d.page_content, metadata=dict(d.metadata)) for d in docs]


def _populate_heading_metadata(metadata: dict, heading: str | None, ctx: dict) -> None:
    """Set the 6 heading-related metadata keys from heading context."""
    if not heading:
        return
    metadata["section"] = heading
    metadata["heading"] = ctx["immediate"][heading]
    metadata["heading_level"] = ctx["level"][heading]
    if ctx["parent"][heading]:
        metadata["parent_section"] = ctx["parent"][heading]
    if ctx["prev"][heading]:
        metadata["prev_heading"] = ctx["prev"][heading]
    if ctx["next"][heading]:
        metadata["next_heading"] = ctx["next"][heading]
    if ctx["siblings"][heading]:
        metadata["sibling_headings"] = ctx["siblings"][heading]


def _normalize_table_content(content: str) -> tuple[str, bool]:
    r"""Detect ``\x00TABLE:`` prefix and strip it; returns (cleaned_content, is_table)."""
    if content.startswith("\x00TABLE:"):
        return content[len("\x00TABLE:") :], True
    return content, False


def _build_section_metadata(
    content: str,
    heading: str | None,
    ctx: dict,
    file_path: Path,
    page_meta: dict | None,
    chunk_index: int,
    total: int,
    is_table: bool,
) -> dict:
    """Assemble the full metadata dict for a single section document."""
    metadata: dict = {"source": file_path.name}
    if page_meta:
        metadata.update(page_meta)
    if is_table:
        metadata["content_type"] = "table"
    metadata["chunk_index"] = chunk_index
    metadata["total_chunks"] = total
    _populate_heading_metadata(metadata, heading, ctx)
    if ctx["toc"]:
        metadata["toc"] = ctx["toc"]
    return metadata


def _heading_context(sections: list[tuple[str | None, str]]) -> dict:
    """Derive level/parent/siblings/prev/next/toc from a list of breadcrumb headings.

    Breadcrumbs already look like "Раздел 1 > Пункт 1.1" (see
    parse_markdown_sections / parse_docx_sections / parse_rtf_sections) — this
    just parses that string structure into per-heading context, once per
    unique breadcrumb, in first-seen document order. Repeated tuples sharing
    the same breadcrumb (e.g. a text segment and a table segment under the
    same heading) all resolve to the same context.
    """
    order: list[str] = []
    seen: set[str] = set()
    for heading, _content in sections:
        if heading and heading not in seen:
            seen.add(heading)
            order.append(heading)

    parts = {h: h.split(" > ") for h in order}
    level = {h: len(parts[h]) for h in order}
    immediate = {h: parts[h][-1] for h in order}
    parent = {h: (" > ".join(parts[h][:-1]) or None) for h in order}

    groups: dict[str | None, list[str]] = {}
    for h in order:
        groups.setdefault(parent[h], []).append(h)

    prev_h: dict[str, str | None] = {}
    next_h: dict[str, str | None] = {}
    siblings: dict[str, list[str]] = {}
    for group in groups.values():
        for i, h in enumerate(group):
            prev_h[h] = immediate[group[i - 1]] if i > 0 else None
            next_h[h] = immediate[group[i + 1]] if i + 1 < len(group) else None
            siblings[h] = [immediate[s] for j, s in enumerate(group) if j != i]

    toc: list[str] = []
    seen_toc: set[str] = set()
    if level:
        top_level = min(level.values())
        for h in order:
            if level[h] == top_level and immediate[h] not in seen_toc:
                seen_toc.add(immediate[h])
                toc.append(immediate[h])

    return {
        "level": level,
        "immediate": immediate,
        "parent": parent,
        "prev": prev_h,
        "next": next_h,
        "siblings": siblings,
        "toc": toc,
    }


class LangchainDocumentParser:
    """Parses files into domain RawDocuments using LangChain infrastructure."""

    def __init__(
        self,
        domain_registry: "DomainProfileRegistry | None" = None,
        domain_settings: "DomainSettingsPort | None" = None,
    ) -> None:
        self._domain_registry = domain_registry
        self._domain_settings = domain_settings

    def parse(self, file_path: Path, *, filename: str | None = None) -> list[RawDocument]:
        docs = self._parse(file_path)
        if filename:
            for doc in docs:
                doc.metadata["source"] = filename
                if doc.metadata.get("_doc_title_fallback"):
                    doc.metadata["doc_title"] = Path(filename).stem
        return docs

    def _parse(self, file_path: Path) -> list[RawDocument]:
        ext = file_path.suffix.lower()

        if ext == ".pdf":
            docs = _lc_to_raw(parse_pdf(file_path))
            return self._apply_doc_level_fallbacks(docs, file_path)

        if ext == ".md":
            page_meta = {}
            title = _extract_md_title(file_path)
            if title:
                page_meta["doc_title"] = title
            md_sections = parse_markdown_sections(file_path)
            docs = self._sections_to_documents(md_sections, file_path, page_meta or None)
            return self._apply_doc_level_fallbacks(docs, file_path)

        if ext in (".docx", ".doc"):
            sections = parse_docx_sections(file_path)
            # Get page metadata from docx page break detection
            _text, page_meta = parse_docx(file_path)
            docs = self._sections_to_documents(sections, file_path, page_meta)
            return self._apply_doc_level_fallbacks(docs, file_path)

        if ext == ".rtf":
            return self._apply_doc_level_fallbacks(self._parse_rtf(file_path), file_path)

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

        docs = [RawDocument(page_content=text, metadata={"source": file_path.name})]
        return self._apply_doc_level_fallbacks(docs, file_path)

    @staticmethod
    def _apply_doc_level_fallbacks(docs: list[RawDocument], file_path: Path) -> list[RawDocument]:
        r"""Fill in doc_title (filename fallback) and doc_type across every chunk.

        Runs after format-specific parsing so a richer title already set by
        a parser (docx core properties, PDF info dict, md's first heading,
        rtf's \\info\\title) is never overwritten — this only fills gaps.
        """
        if not docs:
            return docs

        doc_title = next((d.metadata.get("doc_title") for d in docs if d.metadata.get("doc_title")), None)
        if not doc_title:
            # Fall back to the top segment of the first breadcrumb (e.g. a
            # level-1 "# Title" that has no body content of its own, so it
            # never became a standalone section — still the best title we have).
            first_section = next((d.metadata.get("section") for d in docs if d.metadata.get("section")), None)
            if first_section:
                doc_title = first_section.split(" > ")[0]
        if not doc_title:
            doc_title = file_path.stem
            for doc in docs:
                doc.metadata["_doc_title_fallback"] = True

        sample = f"{doc_title}\n{docs[0].page_content[:DOC_TYPE_SAMPLE_CHARS]}"
        doc_type = classify_doc_type(sample)

        for d in docs:
            d.metadata.setdefault("doc_title", doc_title)
            if doc_type:
                d.metadata.setdefault("doc_type", doc_type)
        return docs

    def _parse_rtf(self, file_path: Path) -> list[RawDocument]:
        """RTF parsing: decree-structured split, generic heuristic split, or flat text.

        Tries the domain-specific decree splitter first (most precise when it
        applies). If that doesn't match, falls back to the generic
        formatting-based section splitter (parse_rtf_sections); if that
        finds no headings either (flat/unstructured RTF), falls back further
        to a single flat document — the same behavior as before this path
        existed, so plain RTF files never regress.
        """
        text, _meta = parse_rtf(file_path)
        if not text or len(text.strip()) < 20:
            raise RuntimeError("Too little text in document")

        decree_docs = self._try_parse_decree_rtf(file_path, text)
        if decree_docs is not None:
            return decree_docs

        try:
            sections = parse_rtf_sections(file_path)
        except Exception:
            log.exception("Structured RTF parsing failed for %s — falling back to flat parse", file_path.name)
            sections = None

        page_meta = {}
        title = _extract_rtf_title(file_path)
        if title:
            page_meta["doc_title"] = title

        if sections and len(sections) > 1:
            return self._sections_to_documents(sections, file_path, page_meta or None)

        metadata = {"source": file_path.name, **page_meta}
        return [RawDocument(page_content=text, metadata=metadata)]

    def _get_decree_profile(self) -> "DecreeDomainProfile | None":
        if self._domain_registry is None or self._domain_settings is None:
            return None
        try:
            profile = self._domain_registry.get("decree")
        except KeyError:
            return None
        return profile if profile.content_boundaries() else None  # type: ignore[return-value]

    def _try_parse_decree_rtf(self, file_path: Path, text: str) -> "list[RawDocument] | None":
        """Decree-structured RTF path; None = fall through to flat RTF parsing.

        Uses the same deterministic fingerprint as document classification,
        applied to the file prefix for speed (see TZ section 5.2).
        """
        profile = self._get_decree_profile()
        if profile is None or not profile.structural_fingerprint(text[:_DECREE_FINGERPRINT_PREFIX_CHARS]):
            return None
        try:
            if self._domain_settings is None:
                return None
            units, doc_metadata = parse_decree_rtf(file_path, profile, self._domain_settings)
        except Exception:
            log.exception("Decree RTF parsing failed for %s — falling back to flat parse", file_path.name)
            return None

        docs = []
        for unit in units:
            if not unit.content.strip():
                continue
            metadata = unit_metadata(unit, profile, {"source": file_path.name, **doc_metadata})
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
        ctx = _heading_context(sections)
        total = len([1 for _h, c in sections if c.strip()])
        docs: list[RawDocument] = []
        chunk_index = 0
        for heading, content in sections:
            if not content.strip():
                continue

            content, is_table = _normalize_table_content(content)
            if not content.strip():
                continue

            chunk_index += 1
            metadata = _build_section_metadata(
                content,
                heading,
                ctx,
                file_path,
                page_meta,
                chunk_index,
                total,
                is_table,
            )
            docs.append(RawDocument(page_content=content, metadata=metadata))

        if not docs:
            raise RuntimeError("Too little text in document")
        attach_table_replacement_scope(docs)
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

    def split(
        self,
        documents: list[RawDocument],
        domain: str = DocDomain.GENERAL.value,
        *,
        profile=None,
        domain_settings=None,
    ) -> list[RawDocument]:
        profile = profile or self._get_profile(domain)
        docs = _raw_to_lc(documents)
        for doc in docs:
            filename = doc.metadata.get("filename") or Path(doc.metadata.get("source", "")).name
            if doc.metadata.pop("_doc_title_fallback", False):
                doc.metadata["doc_title"] = Path(filename).stem
            doc_date = extract_date_from_filename(filename)
            if doc_date:
                doc.metadata.setdefault("doc_date", doc_date)
        return _lc_to_raw(
            _split_documents(
                docs,
                domain=domain,
                profile=profile,
                settings=domain_settings if domain_settings is not None else self._domain_settings,
            )
        )

    def _get_profile(self, domain: str):
        if self._domain_registry is None:
            return None
        try:
            return self._domain_registry.get(domain)
        except KeyError:
            return None
