"""Reconstruct authorized structural conditions for retrieved evidence."""

from __future__ import annotations

from typing import TYPE_CHECKING

from domain.knowledge.chunk_context import content_with_scope


if TYPE_CHECKING:
    from domain.rag.ranking import RetrievalDocument


def content_with_parent_scope(doc: RetrievalDocument, shown_scopes: set[str] | None = None) -> str:
    """Include complete inherited conditions without framework-specific documents."""
    return content_with_scope(doc.metadata, doc.page_content, shown_scopes)
