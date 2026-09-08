"""Lightweight Russian stemmer (suffix stripping) + basic English stemming.

Zero external dependencies. Used by the BM25 tokenizer.
"""

from __future__ import annotations

import re

# Common Russian suffixes to strip for better sparse recall.
# Ordered by length (longest first) to avoid partial matches.
_RU_SUFFIXES = [
    "ости",
    "ость",
    "ений",
    "ение",
    "ания",
    "ями",
    "ого",
    "ать",
    "ить",
    "ыть",
    "ять",
    "ути",
    "яти",
    "ей",
    "ой",
    "ий",
    "ый",
    "ая",
    "яя",
    "ое",
    "ее",
    "ие",
    "ые",
    "ов",
    "ев",
    "ам",
    "ям",
    "ом",
    "ем",
    "ах",
    "ях",
    "ки",
    "ка",
    "ик",
    "ов",
    "ев",
    "ые",
    "ие",
    "ы",
    "и",
    "у",
    "ю",
    "я",
    "е",
    "а",
]

# Sort by length descending for greedy matching
_RU_SUFFIXES.sort(key=len, reverse=True)


def _stem_russian(word: str) -> str:
    """Strip common Russian suffixes to normalize word forms."""
    if len(word) < 5:
        return word
    for suffix in _RU_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def stem_token(token: str) -> str:
    """Apply stemming to a single token."""
    if token.isascii():
        for suffix in [
            "tion", "sion", "ment", "ness", "able", "ible",
            "ful", "less", "ous", "ive",
        ]:
            if token.endswith(suffix) and len(token) - len(suffix) >= 3:
                return token[: -len(suffix)]
        return token
    if re.match(r"[а-яё]", token):
        return _stem_russian(token)
    return token
