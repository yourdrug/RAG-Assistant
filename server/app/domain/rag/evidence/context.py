"""Reconstruct structural conditions without framework-specific documents."""

from domain.knowledge.enums import PageContentType
from domain.rag.ranking import RetrievalDocument


def content_with_parent_scope(doc: RetrievalDocument, shown_scopes: set[str] | None = None) -> str:
    """Include full structural conditions that did not fit inside the chunk."""
    parts = []
    normalized = " ".join(doc.page_content.split())
    scopes = [parent.get("content", "") for parent in doc.metadata.get("parent_units", [])]

    if doc.metadata.get("content_type") == PageContentType.TABLE.value:
        scopes.extend(doc.metadata.get(key, "") for key in ("table_header", "table_row_key"))

    for scope_text in scopes:
        content = scope_text.strip()
        scope = " ".join(content.split())

        if scope and scope not in normalized and (shown_scopes is None or scope not in shown_scopes):
            parts.append(content)
            normalized += " " + scope

        if scope and shown_scopes is not None:
            shown_scopes.add(scope)

    parts.append(doc.page_content)
    return "\n".join(parts)
