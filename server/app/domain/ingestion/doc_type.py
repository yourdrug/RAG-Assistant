"""Document-type classification by keyword heuristics supplied by a domain pack."""

from __future__ import annotations


# (doc_type, keywords) in priority order; the first category with a match wins.
type DocTypeKeywords = tuple[tuple[str, tuple[str, ...]], ...]

DOC_TYPE_SAMPLE_CHARS = 1500


def classify_doc_type(text_sample: str, keywords: DocTypeKeywords) -> str | None:
    """Classify document type by keyword heuristics.

    Checks the first ~1500 chars (title + opening lines) against the pack's
    keyword list. This is deliberately coarse — a cheap signal for
    filtering/ranking, not a legal classification.
    """
    lowered = text_sample.lower()

    for doc_type, words in keywords:
        if any(word in lowered for word in words):
            return doc_type

    return None
