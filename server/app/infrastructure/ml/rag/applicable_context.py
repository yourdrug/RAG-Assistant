"""Recover source-bound scope without inferring it from retrieval relevance."""

import logging
from copy import copy

from domain.rag.evidence.applicability import act_numbers, normalize_act_number
from domain.rag.evidence.scope import (
    table_heading_scope,
    amendment_targets,
    timing_reference,
    timing_definitions,
    timing_appendices,
    timing_clauses,
    needs_timing_scope,
    amendment_requested,
    TIMING_DEFINITIONS_QUERY,
    TIMING_APPENDICES_QUERY,
    TIMING_CLAUSES_QUERY,
    AMENDMENT_ROOT_QUERY,
)
from domain.rag.evidence.queries import listed_goods_polarities, table_scope_confirmed
from domain.knowledge.enums import PageContentType
from domain.knowledge.enums import SearchMode
from domain.rag.evidence.focus import matching_field_rows, requested_fields
from domain.rag.evidence.context import content_with_parent_scope

log = logging.getLogger("default")
SCOPE_WINDOW = 32
MAX_SCOPE_LOOKUPS = 8
MAX_AMENDMENT_LOOKUPS = 3


async def timing_reference_scope(doc, question, chunk_search, user) -> str | None:
    text = content_with_parent_scope(doc)
    reference = timing_reference(text)
    category = listed_goods_polarities(question)
    if not reference or len(category) != 1:
        return None
    metadata = doc.metadata
    options = {
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
    if len(clauses) != 1:
        return None
    return "\n".join(c.content for c in (definitions[0], appendices[0], clauses[0]))


async def enrich_applicable_context(docs, question, chunk_search, user):
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
    for doc, score in docs:
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
        if scope:
            doc = copy(doc)
            doc.metadata = {
                **metadata,
                "parent_units": [*metadata.get("parent_units", []), {"content": scope}],
            }
            if timing_scope:
                doc.metadata["verified_timing_scope"] = timing_scope
        result.append((doc, score))
    return result


async def enrich_amendment_scope(docs, question, chunk_search, user):
    requested = act_numbers(question)
    if not amendment_requested(question):
        return docs
    links: dict[tuple, list[str]] = {}
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
    for doc, score in docs:
        target = links.get((doc.metadata.get("document_id"), doc.metadata.get("act_version_id")))
        if target:
            doc = copy(doc)
            doc.metadata = {**doc.metadata, "verified_amended_acts": target}
        result.append((doc, score))
    return result
