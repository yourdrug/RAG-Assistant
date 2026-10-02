"""Text splitting — merge, split, and route by domain."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from langchain.schema import Document

from infrastructure.ml.ingestion.structural_metadata import unit_metadata
from infrastructure.ml.ingestion.table_chunks import table_document_chunks
from infrastructure.ml.ingestion.page_joining import join_page_word, page_word_evidence
from infrastructure.ml.ingestion.text_chunks import (
    document_slice,
    pack_step_documents,
    text_document_chunks,
)

from config import settings
from domain.value_objects.doc_domain import DocDomain
from domain.value_objects.page_content_type import PageContentType

if TYPE_CHECKING:
    from application.ports.chunk_settings import ChunkSettingsPort
    from domain.domain_profile.settings_port import DomainSettingsPort
    from domain.domain_profile.protocol import DomainProfile

log = logging.getLogger("detailed")

TABLE_BATCH_ROWS = 15  # max data rows per table chunk (excl. header+separator)

GENERAL_SEPARATORS = [
    "\n# ",
    "\n## ",
    "\n### ",
    "\n#### ",
    "\n\n",
    "\n",
    " ",
    "",
]

LEGAL_SEPARATORS = [
    "\nГлава ",
    "\nРаздел ",
    "\nЧасть ",
    "\nСтатья ",
    "\n§ ",
    "\nПункт ",
    "\n\n",
    "\n",
    " ",
    "",
]

# Sentence-aware separators: prefer sentence boundaries, then word, then char
SENTENCE_SEPARATORS = [
    "\n\n",
    "\n",
    ". ",
    "! ",
    "? ",
    ".\n",
    "!\n",
    "?\n",
    "; ",
    ": ",
    " ",
    "",
]

_ARTICLE_RE = re.compile(r"Статья\s+(\d+[\.\d]*)")

# --- Final-chunk metadata enrichment ---------------------------------------
#
# These run on the *final* output chunks — after any structured/legal/char
# splitting has happened — rather than on the pre-split sections upstream in
# langchain_document_parser.py. A section can still get cut into several
# smaller chunks here, so per-chunk facts (its own size, whether it mentions
# a date, its position in the overall sequence) can only be computed
# correctly at this point; computing them earlier would let a stale,
# section-level value leak onto every one of its sub-chunks.

_DATE_RE = re.compile(r"\b(\d{1,2}[./]\d{1,2}[./]\d{2,4}|\d{4}-\d{2}-\d{2})\b")
_DIGIT_RE = re.compile(r"\d")
_LIST_LINE_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_LIST_MIN_LINES = 2
_LIST_MIN_RATIO = 0.6
_SENTENCE_RE = re.compile(r"[.!?…]+[\s\n]+|[.!?…]+$")
_DEFINTION_RE = re.compile(
    r"(?:это|означает|представляет собой|является|определяется как|definition:|means:|is defined as)",
    re.IGNORECASE,
)
_PROCEDURE_RE = re.compile(
    r"(?:шаг\s*\d|пошагов|инструкц|порядок действий|алгоритм|procedure:|step\s*\d|instructions?:)",
    re.IGNORECASE,
)


def _classify_content_shape(text: str) -> str | None:
    """Cheap structural classification: list, definition, procedure, or None."""
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if len(lines) < _LIST_MIN_LINES:
        # Short text: check definition/procedure on full text
        if _DEFINTION_RE.search(text):
            return "definition"
        if _PROCEDURE_RE.search(text):
            return "procedure"
        return None
    list_lines = sum(1 for ln in lines if _LIST_LINE_RE.match(ln))
    if list_lines / len(lines) >= _LIST_MIN_RATIO:
        return "list"
    # Check definition/procedure patterns on multi-line text
    if _DEFINTION_RE.search(text):
        return "definition"
    if _PROCEDURE_RE.search(text):
        return "procedure"
    return None


def _extract_first_sentence(text: str) -> str | None:
    """Return the first sentence of the text (up to 200 chars)."""
    text = text.strip()
    if not text:
        return None
    m = _SENTENCE_RE.search(text)
    if m:
        sentence = text[: m.end()].strip()
        if len(sentence) > 200:
            sentence = sentence[:197] + "..."
        return sentence
    # No sentence boundary found — return first 200 chars
    if len(text) > 200:
        return text[:197] + "..."
    return text


def _enrich_final_chunks(chunks: list[Any]) -> None:
    """Enrich final chunks with derived metadata and re-number.

    Adds char_count, word_count, sentence_count, has_numbers, has_dates,
    extracted_dates, first_sentence, content_type. Re-number chunk_index/
    total_chunks grouped by source file.

    Mutates each Document's metadata in place. chunk_index/total_chunks are
    recomputed here (overwriting any section-level value inherited from
    upstream splitting) because a single section can expand into several
    final chunks — the position that matters for retrieval is the position
    among final chunks, not among pre-split sections.
    """
    by_source: dict[str, list[Any]] = {}
    for c in chunks:
        by_source.setdefault(c.metadata.get("source", ""), []).append(c)

    for group in by_source.values():
        total = len(group)
        for i, c in enumerate(group, start=1):
            c.metadata["chunk_index"] = i
            c.metadata["total_chunks"] = total
            c.metadata["char_count"] = len(c.page_content)
            c.metadata["word_count"] = len(c.page_content.split())
            c.metadata["sentence_count"] = len(_SENTENCE_RE.findall(c.page_content))
            dates = _DATE_RE.findall(c.page_content)
            if dates:
                c.metadata["has_dates"] = True
                c.metadata["extracted_dates"] = dates
            if _DIGIT_RE.search(c.page_content):
                c.metadata["has_numbers"] = True
            first = _extract_first_sentence(c.page_content)
            if first:
                c.metadata["first_sentence"] = first
            if c.metadata.get("content_type") != PageContentType.TABLE.value:
                shape = _classify_content_shape(c.page_content)
                if shape:
                    c.metadata["content_type"] = shape


def merge_pdf_pages(pages: list[Document]) -> list[Document]:
    """Merge consecutive text pages, keeping tables and sections as barriers.

    Page offsets survive until final splitting, where each chunk receives its
    own page range. Sorting within a PDF also restores OCR page order.
    """
    by_source: dict[str, list[Document]] = {}
    for doc in pages:
        by_source.setdefault(doc.metadata.get("source", ""), []).append(doc)
    result: list[Document] = []
    for group in by_source.values():
        if all("page" in doc.metadata for doc in group):
            group = sorted(group, key=lambda doc: doc.metadata["page"])
        pending: list[Document] = []

        for doc in group:
            is_text_page = (
                "page" in doc.metadata and doc.metadata.get("content_type") != PageContentType.TABLE.value
            )
            if not is_text_page:
                result.extend(_merge_page_run(pending))
                pending = []
                result.append(doc)
            else:
                if pending and doc.metadata.get("section") != pending[-1].metadata.get("section"):
                    result.extend(_merge_page_run(pending))
                    pending = []
                pending.append(doc)
        result.extend(_merge_page_run(pending))
    return result


def _merge_page_run(pages: list[Document]) -> list[Document]:
    if not pages:
        return []
    text = ""
    spans: list[tuple[int, int, int]] = []
    evidence = page_word_evidence([doc.page_content for doc in pages])
    for doc in pages:
        content = doc.page_content
        if text:
            joined = join_page_word(text, content, evidence)
            if joined is None:
                text += _page_separator(text, content)
            else:
                text, content = joined
                low, _, page = spans[-1]
                spans[-1] = (low, len(text), page)
        start = len(text)
        text += content
        spans.append((start, len(text), doc.metadata["page"]))
    meta = dict(pages[0].metadata)
    numbers = sorted({page for _, _, page in spans})
    meta.update(page_start=numbers[0], page_end=numbers[-1], pages=numbers, _page_spans=spans)
    return [Document(page_content=text, metadata=meta)]


def _page_separator(before: str, after: str) -> str:
    """Preserve sentence continuity across physical page breaks."""
    structural_start = re.match(
        r"\s*(?:\d+(?:\.\d+)*[.)]\s|[а-я]\)\s|[-*•]\s|#{1,6}\s|Глава\s|Статья\s|Раздел\s)", after
    )
    if structural_start or before.rstrip().endswith((".", "!", "?", ":", ";")):
        return "\n\n"
    return " "


def split_documents(
    docs: list[Document],
    domain: str = DocDomain.GENERAL.value,
    profile: "DomainProfile | None" = None,
    settings: "DomainSettingsPort | None" = None,
    chunk_settings: "ChunkSettingsPort | None" = None,
) -> list[Document]:
    """Split documents into chunks.

    Structured domains (profile with non-empty content_boundaries) are split
    content-based: recursively by the domain's structural boundary hierarchy
    (see split_by_content). max_unit_chars from domain settings is only a
    safety-net that triggers descent to a finer level — it never defines where
    a boundary is.

    Unstructured domains pack complete instruction steps, then use paragraph/
    sentence splitting for oversized blocks. Table row batches retain their
    original position and repeat headers in every domain.
    """
    chunks: list[Document] = []
    for doc in merge_pdf_pages(docs):
        if not doc.page_content.strip():
            continue
        if doc.metadata.get("content_type") == PageContentType.TABLE.value:
            chunks.extend(
                table_document_chunks(
                    doc, settings_chunk_size(domain, settings, chunk_settings), TABLE_BATCH_ROWS
                )
            )
        elif profile is not None and profile.content_boundaries() and settings is not None:
            chunks.extend(_split_structured([doc], profile, settings))
        elif domain == DocDomain.LEGAL.value:
            chunks.extend(split_documents_legal([doc], settings, chunk_settings))
        else:
            chunks.extend(_split_char_general([doc], chunk_settings=chunk_settings))

    # Filter out empty/whitespace-only chunks that cause TEI errors
    before = len(chunks)
    chunks = [c for c in chunks if c.page_content.strip()]

    if len(chunks) < before:
        log.warning("Filtered %d empty chunks during split", before - len(chunks))

    _enrich_final_chunks(chunks)

    log.info(
        "Split %d documents into %d chunks (domain=%s, structured=%s)",
        len(docs),
        len(chunks),
        domain,
        profile is not None and bool(profile.content_boundaries()),
    )
    return chunks


def _split_structured(
    docs: list[Document],
    profile: "DomainProfile",
    domain_settings: "DomainSettingsPort",
) -> list[Document]:
    """Content-based splitting via the profile's structural boundary hierarchy.

    For structured domains (decree, legal), units are first split by content
    boundaries (point/subpoint/sentence). Units still exceeding chunk_size
    are further split using boundary-aware separators from the profile,
    then RecursiveCharacterTextSplitter as final fallback.
    """
    from domain.domain_profile.content_splitter import MIN_STRUCTURAL_CHUNK_CHARS, split_by_content

    max_unit_chars = int(domain_settings.get("max_unit_chars", domain_key=profile.key))
    chunk_size = settings_chunk_size(profile.key, domain_settings)
    default_overlap = (
        settings.legal_chunk_overlap
        if profile.key == DocDomain.LEGAL.value
        else settings.decree_chunk_overlap
    )
    chunk_overlap = _get_domain_setting(
        domain_settings, f"{profile.key}_chunk_overlap", profile.key, default_overlap
    )
    patterns = [level.pattern for level in profile.content_boundaries()]
    result: list[Document] = []
    for doc in docs:
        if "unit_kind" in doc.metadata:
            result.extend(text_document_chunks(doc, chunk_size, chunk_overlap, patterns))
            continue
        units = split_by_content(
            doc.page_content,
            profile.content_boundaries(),
            max_unit_chars,
            min_chunk_chars=MIN_STRUCTURAL_CHUNK_CHARS,
        )
        cursor = 0
        for unit in units:
            unit_doc = document_slice(doc, unit.content, cursor)
            unit_doc.metadata.update(unit_metadata(unit, profile, doc.metadata))
            # The sliced provenance must not be overwritten by document-wide offsets.
            if "_page_spans" in doc.metadata:
                sliced = document_slice(doc, unit.content, cursor)
                unit_doc.metadata["_page_spans"] = sliced.metadata["_page_spans"]
                from infrastructure.ml.ingestion.text_chunks import locate_span

                located = locate_span(doc.page_content, unit.content, cursor)
                if located is not None:
                    cursor = located[1]
            result.extend(text_document_chunks(unit_doc, chunk_size, chunk_overlap, patterns))
    return result


def settings_chunk_size(domain: str, domain_settings=None, chunk_settings=None) -> int:
    """Use the same configured limit in every parser/splitter entry point."""
    global_settings = chunk_settings or settings
    if domain == DocDomain.LEGAL.value:
        default = global_settings.legal_chunk_size
    elif domain == DocDomain.DECREE.value:
        default = settings.decree_chunk_size
    else:
        return global_settings.chunk_size
    return _get_domain_setting(domain_settings, f"{domain}_chunk_size", domain, default)


def _get_domain_setting(settings, key: str, domain_key: str, default: int) -> int:
    """Read a domain-specific setting with fallback to default."""
    if settings is None:
        return default
    try:
        val = int(settings.get(key, domain_key=domain_key))
        return val if val > 0 else default
    except (KeyError, ValueError, TypeError):
        return default


def _split_char_general(
    docs: list[Document],
    chunk_settings: "ChunkSettingsPort | None" = None,
) -> list[Document]:
    """Pack complete steps and split oversized text within the final budget."""
    config = chunk_settings or settings
    chunks: list[Document] = []
    for doc in docs:
        for step in pack_step_documents(doc, config.chunk_size):
            chunks.extend(text_document_chunks(step, config.chunk_size, config.chunk_overlap))
    return chunks


def split_documents_legal(
    docs: list[Document],
    domain_settings: "DomainSettingsPort | None" = None,
    chunk_settings: "ChunkSettingsPort | None" = None,
) -> list[Document]:
    """Split legal documents into chunks with larger size and legal-aware separators.

    Reads legal_chunk_size and legal_chunk_overlap from domain settings if
    available, otherwise falls back to chunk_settings or global defaults.
    """
    if chunk_settings is not None:
        chunk_size = chunk_settings.legal_chunk_size
        chunk_overlap = chunk_settings.legal_chunk_overlap
    else:
        chunk_size = settings.legal_chunk_size
        chunk_overlap = settings.legal_chunk_overlap
    if domain_settings is not None:
        try:
            chunk_size = int(domain_settings.get("legal_chunk_size", "legal"))
        except (KeyError, ValueError):
            pass
        try:
            chunk_overlap = int(domain_settings.get("legal_chunk_overlap", "legal"))
        except (KeyError, ValueError):
            pass

    patterns = [re.compile(re.escape(separator)) for separator in LEGAL_SEPARATORS if separator]
    chunks = [
        chunk for doc in docs for chunk in text_document_chunks(doc, chunk_size, chunk_overlap, patterns)
    ]
    # Filter out empty/whitespace-only chunks that cause TEI errors
    before = len(chunks)
    chunks = [c for c in chunks if c.page_content.strip()]
    if len(chunks) < before:
        log.warning("Filtered %d empty chunks during legal split", before - len(chunks))
    log.info("Split %d legal documents into %d chunks", len(docs), len(chunks))
    return chunks


def extract_article_number(chunk_text: str) -> str | None:
    """Extract article number from chunk text if present."""
    m = _ARTICLE_RE.search(chunk_text)
    return m.group(1) if m else None
