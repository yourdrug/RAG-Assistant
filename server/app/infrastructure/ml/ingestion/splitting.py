"""Text splitting — merge, split, and route by domain."""

from __future__ import annotations

from functools import lru_cache
from itertools import groupby
import logging
import re
from typing import TYPE_CHECKING

from langchain.schema import Document

from config import ChunkingSection, process_section
from domain.ingestion.content_splitter import MIN_STRUCTURAL_CHUNK_CHARS, split_by_content
from domain.ingestion.profiles.decree import DecreeDomainProfile
from domain.ingestion.profiles.legal import LegalDomainProfile
from domain.ingestion.unit_packing import pack_units
from domain.knowledge.enums import DocDomain, PageContentType
from infrastructure.metrics.ingestion import (
    BOUNDARY_CANDIDATES_REJECTED_TOTAL,
    BOUNDARY_NUMBER_GAPS_TOTAL,
    CHUNK_FALLBACK_SPLIT_TOTAL,
)
from infrastructure.ml.ingestion.page_joining import join_page_word, page_word_evidence
from infrastructure.ml.ingestion.structural_metadata import unit_metadata
from infrastructure.ml.ingestion.table_chunks import table_document_chunks
from infrastructure.ml.ingestion.text_chunks import (
    context_prefix,
    document_slice,
    pack_step_documents,
    text_document_chunks,
)
from infrastructure.ml.tokenization import build_token_counter


if TYPE_CHECKING:
    from collections.abc import Sequence

    from domain.ingestion.content_splitter import SplitUnit
    from domain.ingestion.ports import DomainSettingsPort, TokenCounter
    from domain.ingestion.profiles.protocol import DocumentProfile
    from domain.ingestion.unit_packing import PackedUnit
    from domain.rag.ranking import RetrievalDocument

log = logging.getLogger("detailed")

TABLE_BATCH_ROWS = 15  # max data rows per table chunk (excl. header+separator)
MIN_STRUCTURAL_CHUNK_TOKENS = 25

_ARTICLE_RE = re.compile(r"Статья\s+(\d+(?:[.-]\d+)*)")

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

    if len(lines) >= _LIST_MIN_LINES:
        list_lines = sum(1 for ln in lines if _LIST_LINE_RE.match(ln))

        if list_lines / len(lines) >= _LIST_MIN_RATIO:
            return "list"

    for name, pattern in [("definition", _DEFINTION_RE), ("procedure", _PROCEDURE_RE)]:
        if pattern.search(text):
            return name

    return None


def _extract_first_sentence(text: str) -> str | None:
    """Return the first sentence of the text (up to 200 chars)."""
    text = text.strip()

    if not text:
        return None

    m = _SENTENCE_RE.search(text)

    if m:
        sentence = text[: m.end()].strip()

        if len(sentence) > FIRST_SENTENCE_LIMIT:
            sentence = sentence[:197] + "..."

        return sentence

    # No sentence boundary found — return first 200 chars
    if len(text) > FIRST_SENTENCE_LIMIT:
        return text[:197] + "..."

    return text


def _enrich_final_chunks(chunks: Sequence[RetrievalDocument]) -> None:
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
    by_source: dict[str, list[RetrievalDocument]] = {}

    for c in chunks:
        by_source.setdefault(c.metadata.get("source", ""), []).append(c)

    for group in by_source.values():
        total = len(group)

        for i, c in enumerate(group, start=1):
            enrich_chunk_metadata(c, i, total)


def enrich_chunk_metadata(c: RetrievalDocument, i: int, total: int) -> None:
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

    for original_group in by_source.values():
        group = original_group

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
    profile: DocumentProfile | None = None,
    settings: DomainSettingsPort | None = None,
    *,
    token_counter: TokenCounter | None = None,
) -> list[Document]:
    """Split documents into chunks.

    Structured domains (profile with non-empty content_boundaries) are split
    content-based: recursively by the domain's structural boundary hierarchy
    (see split_by_content). The measured body budget triggers descent to a
    finer level; structural markers define the boundaries.

    Unstructured domains pack complete instruction steps, then use paragraph/
    sentence splitting for oversized blocks. Table row batches retain their
    original position and repeat headers in every domain.
    """
    chunker = Chunker(domain, profile, settings, token_counter)
    chunks: list[Document] = []
    table_counts: dict[str, int] = {}

    for original_doc in merge_pdf_pages(docs):
        doc = original_doc

        if not doc.page_content.strip():
            continue

        if doc.metadata.get("content_type") == PageContentType.TABLE.value:
            source = doc.metadata.get("source", "")
            table_counts[source] = table_counts.get(source, 0) + 1
            doc = Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "table_id": f"table-{table_counts[source]}"},
            )
            chunks.extend(
                table_document_chunks(
                    doc,
                    chunker.chunk_size,
                    TABLE_BATCH_ROWS,
                    chunker.measure,
                    target=chunker.target_tokens,
                    on_fallback=chunker.record_fallback,
                )
            )
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
        token_counter: TokenCounter | None = None,
    ) -> None:
        config = process_section(ChunkingSection)
        counter = token_counter or build_token_counter(config.embed_tokenizer_path)
        # Per-document cache bounds repeated tokenization; no private text survives the call.
        self.measure = lru_cache(maxsize=4096)(counter.count)
        self.domain = domain
        self.profile = profile

        if self.profile is None:
            if domain == DocDomain.LEGAL.value:
                self.profile = LegalDomainProfile(domain_settings)
            elif domain == DocDomain.DECREE.value:
                self.profile = DecreeDomainProfile(domain_settings)

        if self.profile and not self.profile.content_boundaries():
            self.profile = None

        self.domain_settings = domain_settings
        key = self.profile.key if self.profile else domain
        default_target = getattr(config, f"{key}_chunk_target_tokens", config.chunk_target_tokens)
        default_max = getattr(config, f"{key}_chunk_max_tokens", config.chunk_max_tokens)
        self.target_tokens = domain_setting(
            domain_settings, f"{key}_chunk_target_tokens", key, default_target
        )
        self.chunk_size = domain_setting(domain_settings, f"{key}_chunk_max_tokens", key, default_max)

        if self.target_tokens > self.chunk_size:
            message = "Target chunk tokens must not exceed maximum chunk tokens"
            raise ValueError(message)

        self.chunk_overlap = 0 if self.profile else config.chunk_overlap_tokens

    def record_fallback(self) -> None:
        CHUNK_FALLBACK_SPLIT_TOTAL.labels(domain=self.domain).inc()

    def record_boundary_event(self, rule: str, position: int) -> None:
        if rule == "number_gap":
            BOUNDARY_NUMBER_GAPS_TOTAL.labels(domain=self.domain).inc()
            log.warning("Skipped structural number (domain=%s, position=%d)", self.domain, position)
        else:
            BOUNDARY_CANDIDATES_REJECTED_TOTAL.labels(domain=self.domain, rule=rule).inc()
            log.debug("Rejected boundary (domain=%s, rule=%s, position=%d)", self.domain, rule, position)

    def text_chunks(self, doc: Document) -> list[Document]:
        return text_document_chunks(
            doc,
            self.chunk_size,
            self.chunk_overlap,
            measure=self.measure,
            target=self.target_tokens,
            on_fallback=self.record_fallback,
        )

    def split(self, doc: Document) -> list[Document]:
        if self.profile is not None:
            return self.split_structured(doc)

        documents = pack_step_documents(doc, self.target_tokens, self.measure)
        return [chunk for item in documents for chunk in self.text_chunks(item)]

    def split_structured(self, doc: Document) -> list[Document]:
        if self.profile is None:
            message = "Structured splitting requires a profile"
            raise RuntimeError(message)

        if "unit_kind" in doc.metadata:
            return self.text_chunks(doc)

        levels = self.profile.content_boundaries()
        units = split_by_content(
            doc.page_content,
            levels,
            self.chunk_size,
            min_chunk_chars=MIN_STRUCTURAL_CHUNK_CHARS,
            unit_budget=lambda unit: self.structured_body_budget(unit, doc),
            unit_label=self.profile.unit_label,
            measure=self.measure,
            on_boundary_event=self.record_boundary_event,
        )
        packed: list[PackedUnit] = []

        for _, siblings in groupby(units, key=lambda unit: unit.parents):
            group = list(siblings)
            # A long condition only reduces its own parent group's body budget.
            budget = min(self.structured_body_budget(unit, doc, target=True) for unit in group)
            packed.extend(
                pack_units(
                    group,
                    budget,
                    self.measure,
                    rank={level.name: index for index, level in enumerate(levels)},
                    min_size=MIN_STRUCTURAL_CHUNK_TOKENS,
                    source=doc.page_content,
                    scope_boundaries=frozenset(level.name for level in levels if level.packing_barrier),
                )
            )

        result: list[Document] = []

        for unit in packed:
            unit_doc = document_slice(doc, unit.content, unit.start)
            spans = unit_doc.metadata.get("_page_spans")
            unit_doc.metadata.update(unit_metadata(unit, self.profile, doc.metadata))

            unit_doc.metadata.update(
                point_numbers=list(unit.point_numbers), boundary_values=list(unit.boundary_values)
            )

            if "_page_spans" in doc.metadata:
                unit_doc.metadata["_page_spans"] = spans

            result.extend(self.text_chunks(unit_doc))

        return result

    def structured_body_budget(self, unit: SplitUnit, doc: Document, *, target: bool = False) -> int:
        """Reserve context for this unit rather than unrelated document scopes."""
        if self.profile is None:
            message = "Structured splitting requires a profile"
            raise RuntimeError(message)

        metadata = unit_metadata(unit, self.profile, doc.metadata)
        # A fitting atom already contains its own heading. Only ancestors and
        # the breadcrumb consume extra space before deciding to descend.
        reserve = self.measure(
            context_prefix(metadata, self.chunk_size, body=unit.content, measure=self.measure)
        )
        limit = self.target_tokens if target else self.chunk_size
        return max(1, int((limit - reserve) * (0.97 if target else 1)))


def domain_setting(
    settings: DomainSettingsPort | None, key: str, domain_key: str, default: int, *, allow_zero: bool = False
) -> int:
    """Read a domain-specific setting with fallback to default."""
    if settings is None:
        return default

    try:
        val = int(settings.get(key, domain_key=domain_key))
    except (KeyError, ValueError, TypeError):
        return default
    else:
        return val if val > 0 or (allow_zero and val == 0) else default


def extract_article_number(chunk_text: str) -> str | None:
    """Extract article number from chunk text if present."""
    m = _ARTICLE_RE.search(chunk_text)
    return m.group(1) if m else None


FIRST_SENTENCE_LIMIT = 200
