"""RAG source extraction — extract and aggregate source metadata from documents."""

from __future__ import annotations

from dataclasses import dataclass, field

from infrastructure.ml.rag.utils import clean_source_name as _clean_source_name


def _merge_list(target: list, source: list) -> None:
    """Append items from source to target, preserving order and avoiding duplicates."""
    for item in source:
        if item not in target:
            target.append(item)


@dataclass
class SourceAccumulator:
    """Aggregated metadata for a single source across all matching documents."""

    pages: set[str] = field(default_factory=set)
    max_score: float = 0.0
    articles: list[str] = field(default_factory=list)
    edited: bool = False
    manual: bool = False
    edited_at: str | None = None
    document_id: int | None = None
    content_hashes: list[str] = field(default_factory=list)
    act_info: dict | None = None
    title: str | None = None
    doc_type: str | None = None
    sections: set[str] = field(default_factory=set)

    def merge(self, other: SourceAccumulator) -> None:
        """Merge another accumulator into this one (take max score, union pages, etc.)."""
        self.pages.update(other.pages)
        if other.max_score > self.max_score:
            self.max_score = other.max_score
        _merge_list(self.articles, other.articles)
        if other.edited:
            self.edited = True
        if other.manual:
            self.manual = True
        if other.edited_at and (self.edited_at is None or other.edited_at > self.edited_at):
            self.edited_at = other.edited_at
        if other.document_id is not None and self.document_id is None:
            self.document_id = other.document_id
        _merge_list(self.content_hashes, other.content_hashes)
        if other.act_info and self.act_info is None:
            self.act_info = other.act_info
        if other.title and self.title is None:
            self.title = other.title
        if other.doc_type and self.doc_type is None:
            self.doc_type = other.doc_type
        self.sections.update(other.sections)


def _collect_source_metadata(doc, score: float | None) -> tuple[str, SourceAccumulator]:
    """Extract metadata from a single document into a SourceAccumulator."""
    src = doc.metadata.get("source", "unknown")
    clean_name = _clean_source_name(src)
    page = doc.metadata.get("page")
    page_start = doc.metadata.get("page_start")
    page_end = doc.metadata.get("page_end")
    pages_list = doc.metadata.get("pages")
    article_number = doc.metadata.get("article_number")
    is_edited = doc.metadata.get("edited", False)
    is_manual = doc.metadata.get("manual", False)
    edited_at = doc.metadata.get("edited_at")
    document_id = doc.metadata.get("document_id")
    ch = doc.metadata.get("content_hash")
    doc_title = doc.metadata.get("doc_title")
    doc_type = doc.metadata.get("doc_type")
    section = doc.metadata.get("section")

    pages_set: set[str] = set()
    if pages_list:
        pages_set.update(pages_list)
    elif page_start is not None and page_end is not None:
        pages_set.update(str(p) for p in range(page_start, page_end + 1))
    elif page is not None:
        pages_set.add(page)

    domain_meta = doc.metadata.get("domain_metadata") or {}
    act_number = domain_meta.get("decree_number") or domain_meta.get("act_number")
    effective_from = doc.metadata.get("effective_from")
    effective_to = doc.metadata.get("effective_to")
    act_info: dict | None = None
    if act_number or effective_from or effective_to:
        act_info = {
            "act_number": act_number,
            "effective_from": effective_from,
            "effective_to": effective_to,
        }

    acc = SourceAccumulator(
        pages=pages_set,
        max_score=score if score is not None else 0.0,
        articles=[article_number] if article_number else [],
        edited=is_edited,
        manual=is_manual,
        edited_at=edited_at,
        document_id=document_id,
        content_hashes=[ch] if ch else [],
        act_info=act_info,
        title=doc_title,
        doc_type=doc_type,
        sections={section} if section else set(),
    )
    return clean_name, acc


def _set_if(entry: dict, key: str, value, condition: bool | None = None) -> None:
    """Add key to entry if value is truthy (or condition is explicitly True)."""
    if condition is None:
        if value:
            entry[key] = value
    elif condition:
        entry[key] = value


def _build_source_entry(src: str, acc: SourceAccumulator) -> dict:
    """Build the entry dict for a single source from its accumulator."""
    entry: dict = {
        "source": src,
        "pages": sorted(acc.pages) if acc.pages else [],
    }
    _set_if(entry, "doc_title", acc.title)
    _set_if(entry, "doc_type", acc.doc_type)
    if acc.sections:
        entry["sections"] = sorted(acc.sections)
    _set_if(entry, "document_id", acc.document_id, condition=acc.document_id is not None)
    _set_if(entry, "content_hashes", acc.content_hashes)
    _set_if(entry, "articles", acc.articles)
    if acc.max_score:
        entry["max_score"] = round(float(acc.max_score), 4)
    for flag in ("edited", "manual"):
        if getattr(acc, flag):
            entry[flag] = True
    _set_if(entry, "edited_at", acc.edited_at)
    _apply_act_info(entry, acc.act_info)
    return entry


def _apply_act_info(entry: dict, act_info: dict | None) -> None:
    """Add act-related fields to entry if present."""
    if not act_info:
        return
    for key in ("act_number", "effective_from", "effective_to"):
        if act_info.get(key):
            entry[key] = act_info[key]


def _filter_sources_by_min_score(
    sources: list[dict],
    min_score: float | None,
) -> list[dict]:
    if min_score is not None and sources:
        filtered = [s for s in sources if s.get("max_score", 0.0) >= min_score]
        if filtered:
            return filtered
        return []
    return sources


def extract_sources(docs, min_score: float | None = None) -> list[dict]:
    """Извлекает метаданные источников для сохранения в БД."""
    accumulators: dict[str, SourceAccumulator] = {}

    for item in docs:
        doc = item[0] if isinstance(item, tuple) else item
        score = item[1] if isinstance(item, tuple) else None
        clean_name, acc = _collect_source_metadata(doc, score)
        if clean_name in accumulators:
            accumulators[clean_name].merge(acc)
        else:
            accumulators[clean_name] = acc

    sources = [_build_source_entry(src, acc) for src, acc in accumulators.items()]

    if accumulators:
        sources.sort(key=lambda s: s.get("max_score", 0.0), reverse=True)

    return _filter_sources_by_min_score(sources, min_score)
