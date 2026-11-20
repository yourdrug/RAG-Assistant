"""Token normalization for BM25: Russian dictionary lemmas plus light English suffix stripping."""

from __future__ import annotations

from functools import cache, lru_cache
import re
from typing import TYPE_CHECKING, Protocol

import pymorphy3


if TYPE_CHECKING:
    from collections.abc import Sequence


MIN_STEM_LENGTH = 3
# Distinct word forms seen by one process stay far below this in practice.
LEMMA_CACHE_SIZE = 200_000

_CYRILLIC_RE = re.compile(r"[а-яё]")
_EN_SUFFIXES = ("tion", "sion", "ment", "ness", "able", "ible", "ful", "less", "ous", "ive")


class WordParse(Protocol):
    normal_form: str


class MorphAnalyzer(Protocol):
    """The part of pymorphy3.MorphAnalyzer used here; the library ships no type hints."""

    def parse(self, word: str) -> Sequence[WordParse]: ...


@cache
def morph_analyzer() -> MorphAnalyzer:
    # Loading the dictionary takes ~0.3 s; done once per process on first use.
    analyzer: MorphAnalyzer = pymorphy3.MorphAnalyzer(lang="ru")
    return analyzer


@lru_cache(maxsize=LEMMA_CACHE_SIZE)
def lemmatize_russian(word: str) -> str:
    """Most probable dictionary form; unknown words get the analyzer's guessed lemma."""
    parses = morph_analyzer().parse(word)
    return parses[0].normal_form.replace("ё", "е") if parses else word


def normalize_token(token: str) -> str:
    """Map a lowercase token to the form shared by its inflections."""
    token = token.replace("ё", "е")

    if token.isascii():
        for suffix in _EN_SUFFIXES:
            if token.endswith(suffix) and len(token) - len(suffix) >= MIN_STEM_LENGTH:
                return token[: -len(suffix)]

        return token

    if _CYRILLIC_RE.match(token):
        return lemmatize_russian(token)

    return token
