"""Recover local interpretation links from authorized, contiguous source chunks."""

from __future__ import annotations

from copy import copy
import logging
from typing import TYPE_CHECKING

from langchain_core.documents import Document

from domain.knowledge.enums import PageContentType
from infrastructure.ml.evidence_profiles.belarus.links import (
    link_modes,
    link_seed_priority,
    procedure_scopes,
    replacement_scopes,
)
from infrastructure.ml.rag.chunk_metadata import chunk_result_metadata


if TYPE_CHECKING:
    from domain.identity.user_context import UserContext
    from domain.knowledge.chunk import ChunkRecord
    from domain.rag.ports import ChunkRetrievalRepository

log = logging.getLogger("default")
LINK_WINDOW = 3
MAX_LINK_LOOKUPS = 8


async def enrich_linked_context(
    docs: list[tuple[Document, float]],
    question: str,
    chunk_search: ChunkRetrievalRepository | None,
    user: UserContext,
) -> list[tuple[Document, float]]:
    """Attach exact atomic scopes; final prompt selection counts their full size.

    The repository enforces ACL. Only the seed's document and act version are
    accepted, including for old indexes without persisted interpretation links.
    Missing neighbors or lookup failures never authorize inferring a link.
    """
    amendment, procedure = link_modes(question)

    if chunk_search is None or not (amendment or procedure):
        return docs

    recovered: dict[tuple[int | None, int | None, int | None], str] = {}
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
        recovered.update(recover_local_scopes(neighbors, doc, is_table=is_table))

    result = []

    for original_doc, score in docs:
        doc = original_doc
        result_identity = (
            doc.metadata.get("document_id"),
            doc.metadata.get("act_version_id"),
            doc.metadata.get("chunk_index"),
        )
        recovered_scope = recovered.get(result_identity)

        doc = attach_recovered_scope(doc, recovered_scope)

        result.append((doc, score))

    return result


def recover_local_scopes(
    neighbors: list[ChunkRecord], doc: Document, *, is_table: bool
) -> dict[tuple[int | None, int | None, int | None], str]:
    metadata = doc.metadata
    document_id = metadata.get("document_id")
    local = [
        Document(page_content=neighbor.content, metadata=chunk_result_metadata(neighbor))
        for neighbor in neighbors
        if neighbor.document_id == document_id and neighbor.act_version_id == metadata.get("act_version_id")
    ]
    local.sort(key=lambda item: item.metadata["chunk_index"])
    scopes = replacement_scopes(local) if is_table else procedure_scopes(local)

    return {(document_id, metadata.get("act_version_id"), index): scope for index, scope in scopes.items()}


def attach_recovered_scope(doc: Document, recovered_scope: str | None) -> Document:
    if recovered_scope:
        doc = copy(doc)
        parents = list(doc.metadata.get("parent_units", []))

        if not any(parent.get("content") == recovered_scope for parent in parents):
            parents.append({"content": recovered_scope})

        doc.metadata = {**doc.metadata, "parent_units": parents}

    return doc
