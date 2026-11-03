"""Bounded row lookup for explicitly named fields missing from vector search."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.documents import Document

from domain.knowledge.enums import PageContentType, SearchMode
from domain.shared.hashing import content_hash
from infrastructure.ml.evidence_profiles.belarus.focus import matching_field_rows, requested_fields
from infrastructure.ml.evidence_profiles.belarus.predicates import table_message_matches
from infrastructure.ml.rag.chunk_metadata import chunk_result_metadata


if TYPE_CHECKING:
    from domain.knowledge.chunk import ChunkRecord
    from domain.rag.context import RagContext
    from domain.rag.ports import ChunkRetrievalRepository

log = logging.getLogger("default")
MAX_FIELD_DOCUMENTS = 3
MAX_FIELD_MATCHES = 32


async def augment_field_candidates(
    question: str, candidates: list[Document], chunk_search: ChunkRetrievalRepository | None, ctx: RagContext
) -> list[Document]:
    """Search only already-retrieved documents through ACL/temporal SQL search.

    Results are candidates, still subject to authoritative document access,
    reranker thresholds and final prompt budgeting. Never fabricate a value.
    """
    fields = requested_fields(question)

    if chunk_search is None or not fields:
        return candidates

    document_ids = list(dict.fromkeys(doc.metadata.get("document_id") for doc in candidates))
    seen = {content_hash(doc.page_content) for doc in candidates}
    result = [doc for doc in candidates if table_message_matches(question, doc.page_content)]

    for number, name in fields[:2]:
        if any(matching_field_rows(doc.page_content, number, name) for doc in result):
            continue

        for document_id in [value for value in document_ids if value is not None][:MAX_FIELD_DOCUMENTS]:
            try:
                matches = await chunk_search.search_substring(
                    query=name,
                    user=ctx.to_user_context(),
                    limit=MAX_FIELD_MATCHES,
                    mode=SearchMode.ICONTAINS.value,
                    document_id=document_id,
                    as_of_date=ctx.as_of_date,
                )
            except Exception:
                log.warning("Field lookup failed for document_id=%s", document_id, exc_info=True)
                continue

            append_matching_fields(
                matches,
                document_id=document_id,
                number=number,
                name=name,
                question=question,
                seen=seen,
                result=result,
            )

    return result


def append_matching_fields(
    matches: list[ChunkRecord],
    *,
    document_id: int,
    number: str,
    name: str,
    question: str,
    seen: set[str],
    result: list[Document],
) -> None:
    for match in matches:
        if match.document_id != document_id or match.content_type != PageContentType.TABLE.value:
            continue

        if not (
            matching_field_rows(match.content, number, name)
            and table_message_matches(question, match.content)
        ):
            continue

        identity = content_hash(match.content)

        if identity not in seen:
            result.append(Document(page_content=match.content, metadata=chunk_result_metadata(match)))
            seen.add(identity)
