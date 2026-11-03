"""Recover source-bound scope without inferring it from retrieval relevance."""

from __future__ import annotations

from copy import copy
import logging
from typing import TYPE_CHECKING, TypedDict

from domain.identity.user_context import UserContext
from domain.knowledge.enums import PageContentType, SearchMode
from domain.rag.evidence.context import content_with_parent_scope
from infrastructure.ml.evidence_profiles.belarus.applicability import act_numbers, normalize_act_number
from infrastructure.ml.evidence_profiles.belarus.focus import matching_field_rows, requested_fields
from infrastructure.ml.evidence_profiles.belarus.predicates import (
    listed_goods_polarities,
    table_scope_confirmed,
)
from infrastructure.ml.evidence_profiles.belarus.scope import (
    AMENDMENT_ROOT_QUERY,
    TIMING_APPENDICES_QUERY,
    TIMING_CLAUSES_QUERY,
    TIMING_DEFINITIONS_QUERY,
    amendment_requested,
    amendment_targets,
    needs_timing_scope,
    table_heading_scope,
    timing_appendices,
    timing_clauses,
    timing_definitions,
    timing_reference,
)


if TYPE_CHECKING:
    from langchain_core.documents import Document

    from domain.identity.user_context import UserContext
    from domain.rag.ports import ChunkRetrievalRepository


class SubstringSearchOptions(TypedDict):
    user: UserContext
    document_id: int
    limit: int
    mode: str


log = logging.getLogger("default")
SCOPE_WINDOW = 32
MAX_SCOPE_LOOKUPS = 8
MAX_AMENDMENT_LOOKUPS = 3


async def timing_reference_scope(
    doc: Document, question: str, chunk_search: ChunkRetrievalRepository, user: UserContext
) -> str | None:
    text = content_with_parent_scope(doc)
    reference = timing_reference(text)
    category = listed_goods_polarities(question)

    if not reference or len(category) != 1:
        return None

    metadata = doc.metadata
    options: SubstringSearchOptions = {
        "user": user,
        "document_id": metadata["document_id"],
        "limit": SCOPE_WINDOW,
        "mode": SearchMode.ICONTAINS.value,
    }
    definitions = timing_definitions(
        await chunk_search.search_substring(query=TIMING_DEFINITIONS_QUERY, **options), metadata, reference
    )

    if len(definitions) != 1:
        return None

    appendices = timing_appendices(
        await chunk_search.search_substring(query=TIMING_APPENDICES_QUERY, **options), metadata
    )

    if len(appendices) != 1:
        return None

    clauses = timing_clauses(
        await chunk_search.search_substring(query=TIMING_CLAUSES_QUERY, **options),
        metadata,
        category,
        appendices[0].chunk_index,
    )

    return (
        "\n".join(c.content for c in (definitions[0], appendices[0], clauses[0]))
        if len(clauses) == 1
        else None
    )


async def enrich_applicable_context(
    docs: list[tuple[Document, float]],
    question: str,
    chunk_search: ChunkRetrievalRepository | None,
    user: UserContext,
) -> list[tuple[Document, float]]:
    """Attach exact table headings or explicit point -> appendix -> category links.

    Every lookup is ACL-scoped. No missing neighbor, version mismatch, or lookup
    failure is permission to fill a scope from the question itself.
    """
    if chunk_search is None:
        return docs

    docs = await enrich_amendment_scope(docs, question, chunk_search, user)
    result = []
    lookups = 0
    fields = requested_fields(question)

    for original_doc, score in docs:
        doc = original_doc
        metadata = doc.metadata
        scope = None
        timing_scope = None

        if metadata.get("document_id") and lookups < MAX_SCOPE_LOOKUPS:
            try:
                if (
                    fields
                    and metadata.get("content_type") == PageContentType.TABLE.value
                    and isinstance(metadata.get("chunk_index"), int)
                    and any(matching_field_rows(doc.page_content, n, name) for n, name in fields)
                    and not table_scope_confirmed(question, content_with_parent_scope(doc))
                ):
                    lookups += 1
                    neighbors = await chunk_search.get_neighbors(
                        metadata["document_id"],
                        metadata["chunk_index"],
                        window=SCOPE_WINDOW,
                        user=user,
                    )
                    scope = table_heading_scope(neighbors, metadata)
                elif needs_timing_scope(question, doc) and lookups + 3 <= MAX_SCOPE_LOOKUPS:
                    lookups += 3
                    timing_scope = await timing_reference_scope(doc, question, chunk_search, user)
                    scope = timing_scope
            except Exception:
                # Enrichment is optional: a failed lookup leaves the scope unconfirmed.
                log.warning(
                    "Applicability scope lookup failed for document_id=%s",
                    metadata["document_id"],
                    exc_info=True,
                )

        doc = attach_applicability_scope(doc, scope, timing_scope)

        result.append((doc, score))

    return result


async def enrich_amendment_scope(
    docs: list[tuple[Document, float]],
    question: str,
    chunk_search: ChunkRetrievalRepository,
    user: UserContext,
) -> list[tuple[Document, float]]:
    requested = act_numbers(question)

    if not amendment_requested(question):
        return docs

    links: dict[tuple[int | None, int | None], list[str]] = {}

    for doc, _ in docs:
        metadata = doc.metadata
        identity = (metadata.get("document_id"), metadata.get("act_version_id"))

        if not all(identity) or identity in links or len(links) >= MAX_AMENDMENT_LOOKUPS:
            continue

        number = metadata.get("act_number")
        actual = (
            {normalize_act_number(str(number))}
            if number
            else act_numbers(metadata.get("doc_title") or metadata.get("source", ""))
        )

        if actual != requested:
            continue

        links[identity] = []
        try:
            roots = await chunk_search.search_substring(
                query=AMENDMENT_ROOT_QUERY,
                user=user,
                document_id=identity[0],
                limit=SCOPE_WINDOW,
                mode=SearchMode.ICONTAINS.value,
            )
        except Exception:
            log.warning("Amendment root lookup failed for document_id=%s", identity[0], exc_info=True)
            continue
        targets = amendment_targets(roots, metadata)

        if len(targets) == 1:
            links[identity] = sorted(targets)

    result = []

    for original_doc, score in docs:
        doc = original_doc
        target = links.get((doc.metadata.get("document_id"), doc.metadata.get("act_version_id")))

        doc = attach_amendment_targets(doc, target)

        result.append((doc, score))

    return result


def attach_applicability_scope(doc: Document, scope: str | None, timing_scope: str | None) -> Document:
    metadata = doc.metadata

    if scope:
        doc = copy(doc)
        doc.metadata = {
            **metadata,
            "parent_units": [*metadata.get("parent_units", []), {"content": scope}],
        }

        if timing_scope:
            doc.metadata["verified_timing_scope"] = timing_scope

    return doc


def attach_amendment_targets(doc: Document, target: list[str] | None) -> Document:
    if not target:
        return doc

    enriched = copy(doc)
    enriched.metadata = {**doc.metadata, "verified_amended_acts": target}
    return enriched
