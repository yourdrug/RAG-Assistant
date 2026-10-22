"""Recover local interpretation links from authorized, contiguous source chunks."""

import logging
from copy import copy

from langchain_core.documents import Document

from domain.knowledge.enums import PageContentType
from infrastructure.ml.rag.chunk_metadata import chunk_result_metadata
from domain.rag.evidence.links import link_modes, link_seed_priority, replacement_scopes, procedure_scopes

log = logging.getLogger("default")
LINK_WINDOW = 3
MAX_LINK_LOOKUPS = 8


async def enrich_linked_context(docs, question, chunk_search, user):
    """Attach exact atomic scopes; final prompt selection counts their full size.

    The repository enforces ACL. Only the seed's document and act version are
    accepted, including for old indexes without persisted interpretation links.
    Missing neighbors or lookup failures never authorize inferring a link.
    """
    amendment, procedure = link_modes(question)

    if chunk_search is None or not (amendment or procedure):
        return docs

    recovered = {}
    lookups = 0
    # Bare amendment tables and actual conditions are more useful seeds than
    # consolidated tables or general definitions when lookup slots are limited.
    ordered = sorted(docs, key=lambda item: link_seed_priority(item[0]))

    for doc, _ in ordered:
        metadata = doc.metadata
        document_id, index = metadata.get("document_id"), metadata.get("chunk_index")
        is_table = metadata.get("content_type") == PageContentType.TABLE.value

        if (
            not document_id
            or not isinstance(index, int)
            or not ((amendment and is_table) or (procedure and not is_table))
        ):
            continue

        identity = (document_id, metadata.get("act_version_id"), index)

        if identity in recovered or lookups >= MAX_LINK_LOOKUPS:
            continue

        lookups += 1
        try:
            neighbors = await chunk_search.get_neighbors(document_id, index, window=LINK_WINDOW, user=user)
        except Exception:
            log.warning("Linked-context lookup failed for document_id=%s", document_id, exc_info=True)
            continue
        local = [
            Document(page_content=neighbor.content, metadata=chunk_result_metadata(neighbor))
            for neighbor in neighbors
            if neighbor.document_id == document_id
            and neighbor.act_version_id == metadata.get("act_version_id")
        ]
        local.sort(key=lambda item: item.metadata["chunk_index"])
        scopes = replacement_scopes(local) if is_table else procedure_scopes(local)

        for chunk_index, scope in scopes.items():
            recovered[(document_id, metadata.get("act_version_id"), chunk_index)] = scope

    result = []

    for doc, score in docs:
        identity = (
            doc.metadata.get("document_id"),
            doc.metadata.get("act_version_id"),
            doc.metadata.get("chunk_index"),
        )
        recovered_scope = recovered.get(identity)

        if recovered_scope:
            doc = copy(doc)
            parents = list(doc.metadata.get("parent_units", []))

            if not any(parent.get("content") == recovered_scope for parent in parents):
                parents.append({"content": recovered_scope})

            doc.metadata = {**doc.metadata, "parent_units": parents}

        result.append((doc, score))

    return result
