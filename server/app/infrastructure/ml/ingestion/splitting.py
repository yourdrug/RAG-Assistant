"""Text splitting — merge, split, and route by domain."""

from __future__ import annotations

from functools import lru_cache
import logging
import re
from typing import TYPE_CHECKING, Any

from langchain.schema import Document

from config import ChunkingSection, process_section
from domain.ingestion.document_tree import DocNode, NodeKind
from domain.ingestion.embed_text import EMBED_PREFIX_KEY, MAX_EMBED_TOKENS, build_embed_text, context_prefix
from domain.ingestion.profiles.contract import ContractDomainProfile
from domain.ingestion.profiles.decree import DecreeDomainProfile
from domain.ingestion.profiles.general import GeneralDomainProfile
from domain.ingestion.profiles.legal import LegalDomainProfile
from domain.ingestion.tree_packer import ChunkDraft, pack_tree
from domain.ingestion.unit_packing import PackedUnit
from domain.knowledge.enums import DocDomain, PageContentType
from infrastructure.metrics.ingestion import (
    BOUNDARY_CANDIDATES_REJECTED_TOTAL,
    BOUNDARY_NUMBER_GAPS_TOTAL,
    CHUNK_FALLBACK_SPLIT_TOTAL,
)
from infrastructure.ml.ingestion.page_joining import join_page_word, page_word_evidence
from infrastructure.ml.ingestion.policy_settings import PolicySettings
from infrastructure.ml.ingestion.structural_metadata import unit_metadata
from infrastructure.ml.ingestion.table_chunks import table_document_chunks
from infrastructure.ml.ingestion.text_chunks import (
    STEP_HEADING_RE,
    document_slice,
    pack_instruction_tree,
    text_tree_chunks,
)
from infrastructure.ml.tokenization import build_token_counter


if TYPE_CHECKING:
    from collections.abc import Sequence

    from domain.ingestion.content_splitter import SplitUnit
    from domain.ingestion.ports import DomainSettingsPort, TokenCounter
    from domain.ingestion.profiles.protocol import DocumentProfile
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


def classify_content_shape(text: str) -> str | None:
    """Classify lists only from explicit line markers."""
    lines = [ln for ln in text.split("\n") if ln.strip()]

    if len(lines) >= _LIST_MIN_LINES:
        list_lines = sum(1 for ln in lines if _LIST_LINE_RE.match(ln))

        if list_lines / len(lines) >= _LIST_MIN_RATIO:
            return "list"

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
        shape = classify_content_shape(c.page_content)

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
    hints: list[dict[str, Any]] = []
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
        hints.extend(
            {
                **hint,
                "start": hint["start"] + start - len(doc.page_content) + len(content),
                "end": hint["end"] + start - len(doc.page_content) + len(content),
            }
            for hint in doc.metadata.get("line_hints", [])
        )

    meta = dict(pages[0].metadata)
    numbers = sorted({page for _, _, page in spans})
    meta["line_hints"] = hints
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
    (see pack_tree). The measured body budget triggers descent to a
    finer level; structural markers define the boundaries.

    Unstructured domains pack complete instruction steps, then use paragraph/
    sentence splitting for oversized blocks. Table row batches retain their
    original position and repeat headers in every domain.
    """
    chunker = TreeChunkingAdapter(domain, profile, settings, token_counter)
    chunks: list[Document] = []
    table_counts: dict[str, int] = {}

    for original_doc in merge_pdf_pages(merge_inferred_sections(docs, chunker)):
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
            tree = DocNode(
                NodeKind.TABLE,
                text=doc.page_content,
                span=(0, len(doc.page_content)),
                hints={"metadata": doc.metadata},
            )
            drafts = pack_tree(
                tree,
                chunker.policy,
                chunker.measure,
                table_packer=lambda node: table_tree_drafts(node, chunker),
            )
            chunks.extend(
                Document(page_content=draft.content, metadata=dict(draft.metadata)) for draft in drafts
            )
        else:
            chunks.extend(chunker.split(doc))

    # Filter out empty/whitespace-only chunks that cause TEI errors
    before = len(chunks)
    chunks = [c for c in chunks if c.page_content.strip()]

    if len(chunks) < before:
        log.warning("Filtered %d empty chunks during split", before - len(chunks))

    for chunk in chunks:
        chunk.metadata.pop("line_hints", None)
        chunk.metadata.pop("inferred_heading", None)
        dense_text = chunk.page_content
        prefix = chunk.metadata.pop("context_prefix", "")
        chunk.page_content = dense_text.removeprefix(prefix)
        # Persisted with the chunk context so manual edits rebuild the same dense formula.
        chunk.metadata[EMBED_PREFIX_KEY] = prefix
        chunk.metadata["embed_text"] = build_embed_text(
            chunk.metadata,
            chunk.page_content,
            max_tokens=MAX_EMBED_TOKENS,
            measure=chunker.measure,
        )

    _enrich_final_chunks(chunks)

    log.info(
        "Split %d documents into %d chunks (domain=%s, structured=%s)",
        len(docs),
        len(chunks),
        domain,
        profile is not None and bool(profile.content_boundaries()),
    )
    return chunks


def table_tree_drafts(tree: DocNode, chunker: TreeChunkingAdapter) -> list[ChunkDraft]:
    documents = table_document_chunks(
        Document(page_content=tree.text, metadata=dict(tree.hints.get("metadata", {}))),
        chunker.chunk_size,
        TABLE_BATCH_ROWS,
        chunker.measure,
        target=chunker.target_tokens,
        on_fallback=chunker.record_fallback,
    )
    return [
        ChunkDraft(
            doc.page_content,
            doc.page_content,
            (),
            tree.span,
            tuple(doc.metadata.get("pages", ())),
            NodeKind.TABLE,
            (),
            None,
            PackedUnit(None, doc.page_content, NodeKind.TABLE.value, start=tree.span[0], end=tree.span[1]),
            doc.metadata,
        )
        for doc in documents
    ]


def merge_inferred_sections(docs: list[Document], chunker: TreeChunkingAdapter) -> list[Document]:
    """Let domain rules refine inferred DOCX headings and pack short guides whole.

    Explicit styles and tables remain boundaries. Short inferred PDF instructions
    retain page spans; larger guides keep the inferred section hierarchy.
    """
    result = []
    pending: list[Document] = []

    for doc in docs:
        inferred = (
            doc.metadata.get("inferred_heading")
            and doc.metadata.get("content_type") != PageContentType.TABLE.value
        )

        if pending and (not inferred or doc.metadata.get("source") != pending[-1].metadata.get("source")):
            result.extend(pack_inferred_run(pending, chunker))
            pending = []

        if inferred:
            pending.append(doc)
        else:
            result.append(doc)

    result.extend(pack_inferred_run(pending, chunker))
    return result


def pack_inferred_run(docs: list[Document], chunker: TreeChunkingAdapter) -> list[Document]:
    if not docs or len(docs) == 1:
        return docs

    pdf_run = all("page" in doc.metadata for doc in docs)

    if pdf_run and (
        chunker.profile is not None
        or not any(STEP_HEADING_RE.match(doc.metadata.get("heading", "")) for doc in docs)
    ):
        return docs

    combined = _merge_page_run(docs)[0] if pdf_run else None
    text = combined.page_content if combined else "\n".join(doc.page_content for doc in docs)
    metadata = dict(combined.metadata if combined else docs[0].metadata)

    if combined:
        metadata.pop("page", None)

    for key in (
        "section",
        "heading",
        "heading_level",
        "parent_section",
        "prev_heading",
        "next_heading",
        "sibling_headings",
    ):
        metadata.pop(key, None)

    metadata["section_headings"] = list(
        dict.fromkeys(doc.metadata["section"] for doc in docs if doc.metadata.get("section"))
    )
    hints: list[dict[str, Any]] = []
    offset = 0

    for doc in docs:
        hints.extend(
            {**hint, "start": hint["start"] + offset, "end": hint["end"] + offset}
            for hint in doc.metadata.get("line_hints", [])
        )
        offset += len(doc.page_content) + 1

    metadata["line_hints"] = combined.metadata["line_hints"] if combined else hints
    prefix = context_prefix(metadata, chunker.chunk_size, body=text, measure=chunker.measure)

    if chunker.profile is None and chunker.measure(prefix + text) > chunker.chunk_size:
        return docs

    return [Document(page_content=text, metadata=metadata)]


class TreeChunkingAdapter:
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
            elif domain == DocDomain.CONTRACT.value:
                self.profile = ContractDomainProfile(domain_settings)
            elif domain == DocDomain.DECREE.value:
                self.profile = DecreeDomainProfile(domain_settings)

        if self.profile and not self.profile.content_boundaries():
            self.profile = None

        self.domain_settings = domain_settings
        policy_profile = self.profile or profile or GeneralDomainProfile()
        self.policy = policy_profile.chunking_policy(PolicySettings(domain_settings, config))

        self.target_tokens = self.policy.target_tokens
        self.chunk_size = self.policy.max_tokens
        self.chunk_overlap = self.policy.overlap_tokens

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
        return text_tree_chunks(
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

        documents = pack_instruction_tree(doc, self.target_tokens, self.measure, policy=self.policy)
        return [chunk for item in documents for chunk in self.text_chunks(item)]

    def split_structured(self, doc: Document) -> list[Document]:
        if self.profile is None:
            message = "Structured splitting requires a profile"
            raise RuntimeError(message)

        if "unit_kind" in doc.metadata:
            return self.text_chunks(doc)

        tree = DocNode(
            NodeKind.DOCUMENT,
            text=doc.page_content,
            span=(0, len(doc.page_content)),
            hints={
                "source_text": doc.page_content,
                "line_hints": doc.metadata.get("line_hints", []),
            },
        )
        tree = self.profile.refine(tree, self.record_boundary_event)
        packed = [
            draft.unit
            for draft in pack_tree(
                tree,
                self.policy,
                self.measure,
                budget=lambda unit, *, target: self.structured_body_budget(unit, doc, target=target),
                split_leaves=False,
            )
        ]

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


def extract_article_number(chunk_text: str) -> str | None:
    """Extract article number from chunk text if present."""
    m = _ARTICLE_RE.search(chunk_text)
    return m.group(1) if m else None


FIRST_SENTENCE_LIMIT = 200
