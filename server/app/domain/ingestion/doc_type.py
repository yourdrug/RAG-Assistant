"""Document-type classification by keyword heuristics."""

from __future__ import annotations


DOC_TYPE_KEYWORDS: list[tuple[str, list[str]]] = [
    ("order", ["приказ", "распоряжение", "постановление"]),
    ("contract", ["договор", "контракт", "соглашение"]),
    ("protocol", ["протокол"]),
    ("instruction", ["инструкция", "регламент", "порядок действий"]),
    ("policy", ["положение \u043e"]),
    ("letter", ["письмо", "уведомление"]),
    ("report", ["отчёт", "отчет"]),
]

DOC_TYPE_SAMPLE_CHARS = 1500


def classify_doc_type(text_sample: str) -> str | None:
    """Classify document type by keyword heuristics.

    Checks the first ~1500 chars (title + opening lines) against a list of
    Russian legal-document keywords. The first category with a match wins.
    This is deliberately coarse — a cheap signal for filtering/ranking, not
    a legal classification.
    """
    lowered = text_sample.lower()

    for doc_type, keywords in DOC_TYPE_KEYWORDS:
        if any(kw in lowered for kw in keywords):
            return doc_type

    return None
