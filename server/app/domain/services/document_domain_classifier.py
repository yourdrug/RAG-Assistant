"""Document domain classifier -- heuristic-based auto-detection of legal vs general documents.

.. deprecated::
    This legacy classifier is superseded by DomainProfileRegistry.classify().
    It is kept as a fallback for environments where the registry is not initialized.
    Will be removed in a future release.
"""

from __future__ import annotations

import re
import warnings

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

    .. deprecated:: Use ``DomainProfileRegistry.classify()`` instead.
    """
    warnings.warn(
        "classify_document_domain() is deprecated; use DomainProfileRegistry.classify()",
        DeprecationWarning,
        stacklevel=2,
    )
    hits = sum(len(re.findall(p, text)) for p in _MARKERS)
    text_len_kb = max(len(text) / 1000, 1)
    density = hits / text_len_kb
    return DocDomain.LEGAL if density >= threshold else DocDomain.GENERAL
