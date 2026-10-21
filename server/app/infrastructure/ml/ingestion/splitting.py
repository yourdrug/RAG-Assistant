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
    locate_span,
    pack_step_documents,
    text_document_chunks,
)

from config import settings
from domain.ingestion.content_splitter import MIN_STRUCTURAL_CHUNK_CHARS, split_by_content
from domain.knowledge.enums import DocDomain
from domain.knowledge.enums import PageContentType

if TYPE_CHECKING:
    from application.chunks import ChunkSettingsPort
    from domain.ingestion.ports import DomainSettingsPort
    from domain.ingestion.profiles.protocol import DocumentProfile

log = logging.getLogger("detailed")

TABLE_BATCH_ROWS = 15  # max data rows per table chunk (excl. header+separator)

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
    profile: "DocumentProfile | None" = None,
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
    chunker = Chunker(domain, profile, settings, chunk_settings)
    chunks: list[Document] = []
    table_counts: dict[str, int] = {}
    for doc in merge_pdf_pages(docs):
        if not doc.page_content.strip():
            continue
        if doc.metadata.get("content_type") == PageContentType.TABLE.value:
            source = doc.metadata.get("source", "")
            table_counts[source] = table_counts.get(source, 0) + 1
            doc = Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "table_id": f"table-{table_counts[source]}"},
            )
            chunks.extend(table_document_chunks(doc, chunker.table_size, TABLE_BATCH_ROWS))
        else:
            chunks.extend(chunker.split(doc))

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


class Chunker:
    """Resolve a document profile's limits once and preserve its splitting policy."""

    def __init__(
        self,
        domain: str,
        profile: DocumentProfile | None = None,
        domain_settings: DomainSettingsPort | None = None,
        chunk_settings: ChunkSettingsPort | None = None,
    ) -> None:
        config = chunk_settings or settings
        self.profile = (
            profile if profile and profile.content_boundaries() and domain_settings is not None else None
        )
        self.domain_settings = domain_settings
        self.table_size = settings_chunk_size(domain, domain_settings, chunk_settings)
        self.patterns: list[re.Pattern[str]] = []
        self.max_unit_chars: int | None = None
        if self.profile is not None and domain_settings is not None:
            key = self.profile.key
            self.chunk_size = (
                self.table_size
                if key == domain and chunk_settings is None
                else settings_chunk_size(key, domain_settings)
            )
            default_overlap = (
                settings.legal_chunk_overlap
                if key == DocDomain.LEGAL.value
                else settings.decree_chunk_overlap
            )
            self.chunk_overlap = domain_setting(domain_settings, f"{key}_chunk_overlap", key, default_overlap)
            self.patterns = [level.pattern for level in self.profile.content_boundaries()]
        elif domain == DocDomain.LEGAL.value:
            self.chunk_size = self.table_size
            self.chunk_overlap = domain_setting(
                domain_settings, "legal_chunk_overlap", domain, config.legal_chunk_overlap, allow_zero=True
            )
            self.patterns = [re.compile(re.escape(separator)) for separator in LEGAL_SEPARATORS if separator]
        else:
            self.chunk_size = config.chunk_size
            self.chunk_overlap = config.chunk_overlap

    def split(self, doc: Document) -> list[Document]:
        if self.profile is not None:
            return self.split_structured(doc)
        documents = [doc] if self.patterns else pack_step_documents(doc, self.chunk_size)
        return [
            chunk
            for item in documents
            for chunk in text_document_chunks(item, self.chunk_size, self.chunk_overlap, self.patterns)
        ]

    def split_structured(self, doc: Document) -> list[Document]:
        if self.profile is None:
            message = "Structured splitting requires a profile"
            raise RuntimeError(message)
        if "unit_kind" in doc.metadata:
            return text_document_chunks(doc, self.chunk_size, self.chunk_overlap, self.patterns)
        if self.max_unit_chars is None:
            if self.domain_settings is None:
                message = "Structured splitting requires domain settings"
                raise RuntimeError(message)
            self.max_unit_chars = int(self.domain_settings.get("max_unit_chars", domain_key=self.profile.key))
        units = split_by_content(
            doc.page_content,
            self.profile.content_boundaries(),
            self.max_unit_chars,
            min_chunk_chars=MIN_STRUCTURAL_CHUNK_CHARS,
        )
        cursor = 0
        result: list[Document] = []
        for unit in units:
            unit_doc = document_slice(doc, unit.content, cursor)
            spans = unit_doc.metadata.get("_page_spans")
            unit_doc.metadata.update(unit_metadata(unit, self.profile, doc.metadata))
            if "_page_spans" in doc.metadata:
                unit_doc.metadata["_page_spans"] = spans
                located = locate_span(doc.page_content, unit.content, cursor)
                if located is not None:
                    cursor = located[1]
            result.extend(text_document_chunks(unit_doc, self.chunk_size, self.chunk_overlap, self.patterns))
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
    return domain_setting(domain_settings, f"{domain}_chunk_size", domain, default)


def domain_setting(settings, key: str, domain_key: str, default: int, *, allow_zero: bool = False) -> int:
    """Read a domain-specific setting with fallback to default."""
    if settings is None:
        return default
    try:
        val = int(settings.get(key, domain_key=domain_key))
        return val if val > 0 or (allow_zero and val == 0) else default
    except (KeyError, ValueError, TypeError):
        return default


def extract_article_number(chunk_text: str) -> str | None:
    """Extract article number from chunk text if present."""
    m = _ARTICLE_RE.search(chunk_text)
    return m.group(1) if m else None
