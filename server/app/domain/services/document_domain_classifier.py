"""Document domain classifier — heuristic-based auto-detection of legal vs general documents.

Pure function with no infrastructure dependencies. Classifies by density of
legal markers (article/chapter references, law citations, contract language)
in the full document text.
"""

from __future__ import annotations

import re

from domain.value_objects.doc_domain import DocDomain

_MARKERS = [
    r"\nСтатья\s+\d+",
    r"\nГлава\s+\d+",
    r"\nРаздел\s+\d+",
    r"\nПункт\s+\d+",
    r"\n\d+\.\d+\.",
    r"Федеральный закон",
    r"ГК РФ",
    r"НК РФ",
    r"настоящ(им|его|ий)\s+(договор|соглашени)",
    r"стороны\s+договорились",
]


def classify_document_domain(text: str, threshold: float = 1.0) -> str:
    """Classify document domain by density of legal markers per 1000 chars.

    Returns DocDomain.LEGAL if marker density >= threshold, else DocDomain.GENERAL.
    Safe default is "general" — worst case a legal document doesn't get a boost.
    """
    hits = sum(len(re.findall(p, text)) for p in _MARKERS)
    text_len_kb = max(len(text) / 1000, 1)
    density = hits / text_len_kb
    return DocDomain.LEGAL if density >= threshold else DocDomain.GENERAL
