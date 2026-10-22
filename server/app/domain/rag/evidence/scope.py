"""Require exact source identity and contiguous evidence for recovered scope."""

import re
from typing import Any

from domain.knowledge.chunk import ChunkRecord
from domain.rag.evidence.applicability import act_numbers, normalize_act_number, timing_question
from domain.rag.evidence.context import content_with_parent_scope
from domain.rag.evidence.queries import listed_goods_polarities
from domain.rag.retrieval import RetrievalDocument


def same_version(chunk: ChunkRecord, metadata: dict[str, Any]) -> bool:
    return chunk.document_id == metadata.get("document_id") and chunk.act_version_id == metadata.get(
        "act_version_id"
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


def timing_appendices(chunks: list[ChunkRecord], metadata: dict[str, Any]) -> list[ChunkRecord]:
    return [
        c
        for c in chunks
        if same_version(c, metadata)
        and re.search(r"Приложение\s+к постановлению", c.content, re.IGNORECASE)
        and "СОСТАВ" in c.content.upper()
        and "СВЕДЕНИЙ" in c.content.upper()
    ]


def timing_clauses(
    chunks: list[ChunkRecord], metadata: dict[str, Any], category: set[bool], appendix_index: int
) -> list[ChunkRecord]:
    return [
        c
        for c in chunks
        if same_version(c, metadata)
        and c.chunk_index > appendix_index
        and listed_goods_polarities(c.content) == category
        and re.search(r"При передаче", c.content, re.IGNORECASE)
    ]


def needs_timing_scope(question: str, doc: RetrievalDocument) -> bool:
    return bool(
        timing_question(question)
        and not listed_goods_polarities(content_with_parent_scope(doc))
        and re.search(r"сведения, указанные в пункте \d+ настоящего постановления", doc.page_content)
    )


def amendment_requested(question: str) -> bool:
    return bool(act_numbers(question) and re.search(r"измен|постановлением|замен", question, re.IGNORECASE))
