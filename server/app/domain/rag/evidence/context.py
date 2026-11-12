"""Reconstruct structural conditions without framework-specific documents."""

from domain.knowledge.enums import PageContentType
from domain.rag.ranking import RetrievalDocument


def content_with_parent_scope(doc: RetrievalDocument, shown_scopes: set[str] | None = None) -> str:
    """Include full structural conditions that did not fit inside the chunk."""
    parts = []
    normalized = " ".join(doc.page_content.split())
    scopes = [
        (parent.get("heading") or "", parent.get("content") or "")
        for parent in doc.metadata.get("parent_units", [])
    ]

    if doc.metadata.get("content_type") == PageContentType.TABLE.value:
        scopes.extend(("", doc.metadata.get(key) or "") for key in ("table_header", "table_row_key"))

    for heading, scope_text in scopes:
        content = scope_text.strip()
        scope = " ".join(content.split())

        if scope and scope not in normalized and (shown_scopes is None or scope not in shown_scopes):
            missing = uncovered_scope(heading, content, normalized)

            if missing:
                parts.append(missing)
                normalized += " " + " ".join(missing.split())

        if scope and shown_scopes is not None:
            shown_scopes.add(scope)

    parts.append(doc.page_content)
    return "\n".join(parts)


def uncovered_scope(heading: str, content: str, normalized: str) -> str:
    """Return the part of a parent scope absent from the text.

    Context prefixes carry the parent heading in the section path and its
    introduction on a separate line, so the scope is present but not verbatim.
    """
    title = " ".join(heading.split())
    scope = " ".join(content.split())

    if not title or not scope.startswith(title):
        return content

    introduction = scope[len(title) :].strip()

    if not introduction or introduction not in normalized:
        return content

    return "" if title in normalized else heading.strip()
