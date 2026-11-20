"""Require exact source identity and contiguous evidence for recovered scope."""

import re
from typing import Any

from domain.domain_packs.belarus_law.acts.edition_metadata import edition_id
from domain.knowledge.chunk import ChunkRecord
from domain.knowledge.pack_metadata import pack_metadata_of
from domain.rag.evidence.context import content_with_parent_scope
from domain.rag.ranking import RetrievalDocument
from infrastructure.domain_packs.belarus_law.evidence.applicability import (
    act_numbers,
    normalize_act_number,
    timing_question,
)
from infrastructure.domain_packs.belarus_law.evidence.predicates import listed_goods_polarities


def same_version(chunk: ChunkRecord, metadata: dict[str, Any]) -> bool:
    return chunk.document_id == metadata.get("document_id") and edition_id(chunk.pack_metadata) == edition_id(
        pack_metadata_of(metadata)
    )


def table_heading_scope(neighbors: list[ChunkRecord], metadata: dict[str, Any]) -> str | None:
    local = sorted((c for c in neighbors if same_version(c, metadata)), key=lambda c: c.chunk_index)
    index = metadata["chunk_index"]
    headings = [c for c in local if c.chunk_index < index and re.match(r"Таблица\s+\d", c.content)]

    if not headings:
        return None

    heading = headings[-1]
    chain = [c for c in local if heading.chunk_index <= c.chunk_index <= index]

    if [c.chunk_index for c in chain] != list(range(heading.chunk_index, index + 1)):
        return None

    table_id = metadata.get("table_id")

    if not table_id or any(c.context_metadata.get("table_id") != table_id for c in chain[1:]):
        return None

    return heading.content


def amendment_targets(roots: list[ChunkRecord], metadata: dict[str, Any]) -> set[str]:
    targets = set()

    for chunk in roots:
        if not same_version(chunk, metadata):
            continue

        match = re.search(
            r"Внести[\s\S]*?постановлен\w+[\s\S]*?(?:\bN|№)\s*(\d+(?:[/\-]\d+)*),?\s*следующие изменения",
            chunk.content,
            re.IGNORECASE,
        )

        if match:
            targets.add(normalize_act_number(match[1]))

    return targets


TIMING_DEFINITIONS_QUERY = "согласно приложению"
TIMING_APPENDICES_QUERY = "Приложение"
TIMING_CLAUSES_QUERY = "включен"
AMENDMENT_ROOT_QUERY = "следующие изменения"


def timing_reference(text: str) -> str | None:
    match = re.search(r"сведения, указанные в пункте (\d+) настоящего постановления", text, re.IGNORECASE)
    return match[1] if match else None


def timing_definitions(
    chunks: list[ChunkRecord], metadata: dict[str, Any], reference: str
) -> list[ChunkRecord]:
    return [
        c
        for c in chunks
        if same_version(c, metadata) and re.search(rf"^{reference}\.\s", c.content, re.MULTILINE)
    ]


def appendix_title(chunk: ChunkRecord) -> bool:
    return bool(re.search(r"Приложение\s+к постановлению", chunk.content, re.IGNORECASE))


def composition_title(chunk: ChunkRecord) -> bool:
    return "СОСТАВ" in chunk.content.upper() and "СВЕДЕНИЙ" in chunk.content.upper()


def timing_appendices(chunks: list[ChunkRecord], metadata: dict[str, Any]) -> list[ChunkRecord]:
    return [c for c in chunks if same_version(c, metadata) and appendix_title(c) and composition_title(c)]


def timing_appendix_titles(chunks: list[ChunkRecord], metadata: dict[str, Any]) -> list[ChunkRecord]:
    return [c for c in chunks if same_version(c, metadata) and appendix_title(c) and not composition_title(c)]


def split_timing_appendix(
    title: ChunkRecord, neighbors: list[ChunkRecord], metadata: dict[str, Any]
) -> ChunkRecord | None:
    """Return the chunk that continues an appendix heading split by chunking.

    Only the immediately following chunk of the same document version counts,
    and it must open with the composition heading rather than mention it later.
    """
    following = [c for c in neighbors if same_version(c, metadata) and c.chunk_index == title.chunk_index + 1]

    if len(following) != 1:
        return None

    heading = re.split(r"^\d+\.\s", following[0].content, maxsplit=1, flags=re.MULTILINE)[0]
    return following[0] if re.search(r"\bСОСТАВ\s+СВЕДЕНИЙ\b", heading) else None


def timing_clauses(
    chunks: list[ChunkRecord], metadata: dict[str, Any], category: set[bool], appendix_index: int
) -> list[ChunkRecord]:
    """Return appendix clauses of the requested goods category, in source order.

    Several clauses may share the category; any appendix clause naming another
    category makes the appendix scope ambiguous, so nothing is returned.
    """
    local = [c for c in chunks if same_version(c, metadata) and c.chunk_index > appendix_index]

    if any(listed_goods_polarities(c.content) - category for c in local):
        return []

    return sorted(
        (
            c
            for c in local
            if listed_goods_polarities(c.content) == category
            and re.search(r"При передаче", c.content, re.IGNORECASE)
        ),
        key=lambda c: c.chunk_index,
    )


def needs_timing_scope(question: str, doc: RetrievalDocument) -> bool:
    return bool(
        timing_question(question)
        and not listed_goods_polarities(content_with_parent_scope(doc))
        and re.search(r"сведения, указанные в пункте \d+ настоящего постановления", doc.page_content)
    )


def amendment_requested(question: str) -> bool:
    return bool(act_numbers(question) and re.search(r"измен|постановлением|замен", question, re.IGNORECASE))
