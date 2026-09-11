"""Text splitting — merge, split, and route by domain."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import settings
from domain.value_objects.doc_domain import DocDomain
from domain.value_objects.page_content_type import PageContentType

if TYPE_CHECKING:
    from application.ports.domain_settings import DomainSettingsPort
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


def _enrich_final_chunks(chunks: list[Document]) -> None:
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
    by_source: dict[str, list[Document]] = {}
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
    """Merge per-page Documents from the same source into a single Document.

    Preserves page range in metadata (page_start, page_end, pages list).
    Prevents the splitter from breaking text at page boundaries.
    """
    if len(pages) <= 1:
        return pages

    # Group by source
    by_source: dict[str, list[Document]] = {}
    for p in pages:
        src = p.metadata.get("source", "")
        by_source.setdefault(src, []).append(p)

    merged = []
    for _src, group in by_source.items():
        merged_text = "\n\n".join(p.page_content for p in group)
        has_pages = any("page" in p.metadata for p in group)
        meta = {**group[0].metadata}
        if has_pages:
            page_nums = sorted(p.metadata.get("page", i + 1) for i, p in enumerate(group))
            meta["page_start"] = page_nums[0]
            meta["page_end"] = page_nums[-1]
            meta["pages"] = page_nums
        merged.append(Document(page_content=merged_text, metadata=meta))
    return merged


def _split_table_into_batches(table_doc: Document, max_rows: int) -> list[Document]:
    """Split a large markdown table into batched chunks with repeated headers.

    Each batch contains at most *max_rows* data rows plus the header and
    separator line repeated.  Small tables (<= max_rows) are returned as-is.
    """
    lines = table_doc.page_content.split("\n")
    if len(lines) < 3:
        return [table_doc]

    header_line = lines[0]
    separator_line = lines[1]
    data_lines = lines[2:]

    if len(data_lines) <= max_rows:
        return [table_doc]

    batches: list[Document] = []
    for i in range(0, len(data_lines), max_rows):
        batch_rows = data_lines[i : i + max_rows]
        batch_text = "\n".join([header_line, separator_line] + batch_rows)
        batches.append(Document(page_content=batch_text, metadata={**table_doc.metadata}))
    return batches


def split_documents(
    docs: list[Document],
    domain: str = DocDomain.GENERAL.value,
    profile: "DomainProfile | None" = None,
    settings: "DomainSettingsPort | None" = None,
) -> list[Document]:
    """Split documents into chunks.

    Structured domains (profile with non-empty content_boundaries) are split
    content-based: recursively by the domain's structural boundary hierarchy
    (see split_by_content). max_unit_chars from domain settings is only a
    safety-net that triggers descent to a finer level — it never defines where
    a boundary is.

    Unstructured domains use RecursiveCharacterTextSplitter with domain-aware
    separators. Table chunks (content_type=table) are split into row batches
    (TABLE_BATCH_ROWS per chunk) with header repetition in both paths.
    """
    tables = [d for d in docs if d.metadata.get("content_type") == PageContentType.TABLE.value]
    text_docs = [d for d in docs if d.metadata.get("content_type") != PageContentType.TABLE.value]

    if profile is not None and profile.content_boundaries() and settings is not None:
        chunks = _split_structured(text_docs, profile, settings)
    elif domain == DocDomain.LEGAL.value:
        chunks = split_documents_legal(text_docs, settings)
    else:
        chunks = _split_char_general(text_docs)

    # Split large tables into row batches with repeated headers
    for table_doc in tables:
        chunks.extend(_split_table_into_batches(table_doc, TABLE_BATCH_ROWS))

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
    settings: "DomainSettingsPort",
) -> list[Document]:
    """Content-based splitting via the profile's structural boundary hierarchy.

    For structured domains (decree, legal), units are first split by content
    boundaries (point/subpoint/sentence). Units still exceeding chunk_size
    are further split using boundary-aware separators from the profile,
    then RecursiveCharacterTextSplitter as final fallback.
    """
    from domain.domain_profile.content_splitter import split_by_content
    from domain.domain_profile.protocol import refs_to_metadata

    max_unit_chars = int(settings.get("max_unit_chars", domain_key=profile.key))
    chunk_size = _get_domain_setting(settings, f"{profile.key}_chunk_size", profile.key, default=1200)
    chunk_overlap = _get_domain_setting(settings, f"{profile.key}_chunk_overlap", profile.key, default=200)

    # Extract boundary regex patterns from profile for overflow splitting
    boundary_patterns = [bl.pattern for bl in profile.content_boundaries()]

    result: list[Document] = []
    for doc in docs:
        for unit in split_by_content(
            doc.page_content, profile.content_boundaries(), max_unit_chars, min_chunk_chars=100
        ):
            meta = {**doc.metadata, "unit_kind": unit.unit_kind}
            if unit.heading:
                meta["section"] = unit.heading
            if unit.boundary_value:
                meta[f"{unit.unit_kind}_number"] = unit.boundary_value
            unit_refs = profile.extract_references(unit.content)
            if unit_refs:
                meta["domain_metadata"] = refs_to_metadata(unit_refs)

            if len(unit.content) > chunk_size:
                overflow_chunks = _split_overflow(
                    unit.content, chunk_size, chunk_overlap, boundary_patterns
                )
                if len(overflow_chunks) > 1:
                    # Extract first meaningful line as context prefix for non-first chunks
                    first_line = unit.content.split("\n", 1)[0].strip()
                    if len(first_line) > 10:
                        prefix = first_line + "\n"
                        overflow_chunks = [
                            overflow_chunks[0],
                            *[prefix + c for c in overflow_chunks[1:]],
                        ]
                for overflow_text in overflow_chunks:
                    result.append(Document(page_content=overflow_text, metadata=dict(meta)))
            else:
                result.append(Document(page_content=unit.content, metadata=meta))
    return result


def _split_overflow(  # noqa: C901
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    boundary_patterns: list | None = None,
) -> list[str]:
    """Split oversized text with overlap as a safety net.

    When boundary_patterns are provided (for structured domains), first split
    at structural boundaries and merge fragments to respect chunk_size. This
    preserves logical unit integrity — rules for different categories stay
    together. Falls back to RecursiveCharacterTextSplitter for any remaining
    oversized fragments.
    """
    if boundary_patterns:
        # Step 1: Split at all boundary patterns
        fragments = [text]
        for pattern in boundary_patterns:
            new_fragments = []
            for frag in fragments:
                new_fragments.extend(re.split(pattern, frag))
            fragments = [f for f in fragments if f.strip()]

        # Step 2: Merge fragments respecting chunk_size (with overlap)
        merged: list[str] = []
        current = ""
        for frag in fragments:
            frag = frag.strip()
            if not frag:
                continue
            if not current:
                current = frag
            elif len(current) + len(frag) + 1 <= chunk_size:
                current = current + "\n" + frag
            else:
                merged.append(current)
                # Overlap: keep tail of current chunk
                if chunk_overlap > 0 and len(current) > chunk_overlap:
                    current = current[-chunk_overlap:] + "\n" + frag
                else:
                    current = frag
        if current.strip():
            merged.append(current)

        # Step 3: If any fragment still exceeds chunk_size, use generic splitter
        result = []
        for chunk in merged:
            if len(chunk) > chunk_size:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    length_function=len,
                    separators=["\n\n", "\n", ". ", " ", ""],
                )
                result.extend(splitter.split_text(chunk))
            else:
                result.append(chunk)
        return result

    # Fallback: no boundary patterns — use generic splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def _get_domain_setting(settings, key: str, domain_key: str, default: int) -> int:
    """Read a domain-specific setting with fallback to default."""
    if settings is None:
        return default
    try:
        val = int(settings.get(key, domain_key=domain_key))
        return val if val > 0 else default
    except (KeyError, ValueError, TypeError):
        return default


def _split_char_general(docs: list[Document]) -> list[Document]:
    """Char-based splitting for unstructured domains (paragraph-aware separators)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        separators=GENERAL_SEPARATORS,
    )
    return splitter.split_documents(docs)


def split_documents_legal(
    docs: list[Document],
    domain_settings: "DomainSettingsPort | None" = None,
) -> list[Document]:
    """Split legal documents into chunks with larger size and legal-aware separators.

    Reads legal_chunk_size and legal_chunk_overlap from domain settings if
    available, otherwise falls back to defaults (1000 / 250).
    """
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

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=LEGAL_SEPARATORS,
    )
    chunks = splitter.split_documents(docs)
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
