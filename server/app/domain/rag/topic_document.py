"""Which retrieved document is about the subject of a question nothing relevant was found for.

When the reranker rejects every candidate, the question may still ask for a value
a document on its subject does not set («какой лимит стоимости гостиницы» against a
travel approval procedure). Only a document whose title or opening names the subject
qualifies, and only when it is read whole can its silence be stated.
"""

from __future__ import annotations

import re


# Short words («лимит», «срок») name values, not subjects; long ones name the subject.
MIN_SUBJECT_WORD = 7
SUBJECT_STEM = 6


def subject_stems(question: str) -> frozenset[str]:
    return frozenset(
        word[:SUBJECT_STEM]
        for word in re.findall(r"[а-яё]+", question.casefold())
        if len(word) >= MIN_SUBJECT_WORD
    )


def names_subject(heading: str, stems: frozenset[str]) -> bool:
    """Whether a document title or opening names one of the question's subject words."""
    words = {
        word[:SUBJECT_STEM]
        for word in re.findall(r"[а-яё]+", heading.casefold())
        if len(word) >= SUBJECT_STEM
    }
    return bool(stems & words)


def complete_indexes(indexes: list[int], window: int) -> bool:
    """Whether indexes read from 0 to ``window`` cover a whole document.

    They must be contiguous from 0, and index ``window`` must be absent: present,
    it shows the document goes on beyond what was read.
    """
    return 0 < len(indexes) <= window and sorted(indexes) == list(range(len(indexes)))
