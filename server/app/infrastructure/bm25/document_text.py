"""Choose lexical search text while leaving full-content identity unchanged."""

from domain.rag.ranking import RetrievalDocument


def bm25_document_text(doc: RetrievalDocument) -> str:
    """Use the source body; ignore stale derived metadata after manual edits."""
    prefix = doc.metadata.get("context_prefix") or ""
    body = doc.metadata.get("body_text")

    if isinstance(body, str) and prefix + body == doc.page_content:
        return body

    return doc.page_content.removeprefix(prefix)
