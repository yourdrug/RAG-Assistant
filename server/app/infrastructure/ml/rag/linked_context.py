"""Recover local interpretation links from authorized, contiguous source chunks."""

import logging
import re
from copy import copy

from langchain_core.documents import Document

from domain.value_objects.page_content_type import PageContentType
from infrastructure.ml.rag.chunk_metadata import chunk_result_metadata

log = logging.getLogger("default")
LINK_WINDOW = 3
MAX_LINK_LOOKUPS = 8
STAGE_RE = re.compile(r"^При\s+(?:передаче|отгрузке|реализации)\b", re.IGNORECASE)


def chunk_body(doc) -> str:
    """Ignore repeated structural headings when finding a sentence's condition."""
    lines = doc.page_content.splitlines()
    parents = {parent.get("content", "").strip() for parent in doc.metadata.get("parent_units", [])}
    return "\n".join(
        line for line in lines if not line.startswith("[Раздел:") and line.strip() not in parents
    )


def replacement_scopes(docs) -> dict[int, str]:
    scopes = {}
    for target, old, bridge, new in zip(docs, docs[1:], docs[2:], docs[3:], strict=False):
        if not re.search(r"в\s+таблице\s+\d+(?:\.\d+)*", target.page_content, re.IGNORECASE):
            continue
        if not re.search(r"\bпозицию\s*:?\s*$", target.page_content):
            continue
        if not re.fullmatch(r"\s*заменить\s+позицией\s*:?\s*", bridge.page_content, re.IGNORECASE):
            continue
        if any(doc.metadata.get("content_type") != PageContentType.TABLE.value for doc in (old, new)):
            continue
        # Links are source adjacency, never relevance/prompt ordering.
        indices = [doc.metadata["chunk_index"] for doc in (target, old, bridge, new)]
        if indices != list(range(indices[0], indices[0] + 4)):
            continue
        start = list(re.finditer(r"в\s+таблице\s+\d+(?:\.\d+)*", target.page_content, re.IGNORECASE))[
            -1
        ].start()
        scope = "\n".join(
            [target.page_content[start:], old.page_content, bridge.page_content, new.page_content]
        )
        scopes[old.metadata["chunk_index"]] = scope
        scopes[new.metadata["chunk_index"]] = scope
    return scopes


def procedure_scopes(docs) -> dict[int, str]:
    scopes = {}
    for previous, current in zip(docs, docs[1:], strict=False):
        if current.metadata["chunk_index"] != previous.metadata["chunk_index"] + 1:
            continue
        if current.metadata.get("section") != previous.metadata.get("section"):
            continue
        if any(STAGE_RE.match(line) for line in chunk_body(current).splitlines()):
            continue
        condition = next((line for line in chunk_body(previous).splitlines() if STAGE_RE.match(line)), None)
        continuation = next(
            (line for line in chunk_body(current).splitlines() if re.match(r"В накладной\s", line)), None
        )
        if condition and continuation:
            scope = condition + "\n" + continuation
            scopes[previous.metadata["chunk_index"]] = scope
            scopes[current.metadata["chunk_index"]] = scope
    return scopes


async def enrich_linked_context(docs, question, chunk_search, user):
    """Attach exact atomic scopes; final prompt selection counts their full size.

    The repository enforces ACL. Only the seed's document and act version are
    accepted, including for old indexes without persisted interpretation links.
    Missing neighbors or lookup failures never authorize inferring a link.
    """
    amendment = bool(re.search(r"измен|замен", question, re.IGNORECASE))
    procedure = bool(re.search(r"поряд|этап|последующ", question, re.IGNORECASE))
    if chunk_search is None or not (amendment or procedure):
        return docs
    recovered = {}
    lookups = 0
    # Bare amendment tables and actual conditions are more useful seeds than
    # consolidated tables or general definitions when lookup slots are limited.
    ordered = sorted(
        docs,
        key=lambda item: bool(item[0].metadata.get("table_header"))
        if item[0].metadata.get("content_type") == PageContentType.TABLE.value
        else not any(
            STAGE_RE.match(line) or line.startswith("В накладной ")
            for line in chunk_body(item[0]).splitlines()
        ),
    )
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
        scope = recovered.get(identity)
        if scope:
            doc = copy(doc)
            parents = list(doc.metadata.get("parent_units", []))
            if not any(parent.get("content") == scope for parent in parents):
                parents.append({"content": scope})
            doc.metadata = {**doc.metadata, "parent_units": parents}
        result.append((doc, score))
    return result
