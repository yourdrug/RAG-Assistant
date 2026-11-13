"""Lexical search uses the persisted source body."""

from domain.rag.ranking import RetrievalDocument


def bm25_document_text(doc: RetrievalDocument) -> str:
    """Dense context never contributes lexical terms."""
    return doc.page_content
