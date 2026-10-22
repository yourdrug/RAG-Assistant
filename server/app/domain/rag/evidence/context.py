"""Reconstruct structural conditions without framework-specific documents."""

from domain.knowledge.enums import PageContentType
from domain.rag.retrieval import RetrievalDocument


def content_with_parent_scope(doc: RetrievalDocument) -> str:
    """Include full structural conditions that did not fit inside the chunk."""
    parts = []
    normalized = " ".join(doc.page_content.split())
    scopes = [parent.get("content", "") for parent in doc.metadata.get("parent_units", [])]

    if doc.metadata.get("content_type") == PageContentType.TABLE.value:
        scopes.extend(doc.metadata.get(key, "") for key in ("table_header", "table_row_key"))

    for scope_text in scopes:
        content = scope_text.strip()
        scope = " ".join(content.split())

        if scope and scope not in normalized:
            parts.append(content)
            normalized += " " + scope

    parts.append(doc.page_content)
    return "\n".join(parts)
