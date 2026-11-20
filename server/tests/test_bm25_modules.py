"""Token normalization tests for the BM25 tokenizer used by Qdrant sparse vectors."""

import pytest

from infrastructure.bm25.lemmatizer import normalize_token


class TestNormalizeToken:
    @pytest.mark.parametrize(
        "word,lemma",
        [
            ("маркировки", "маркировка"),
            ("документами", "документ"),
            ("налоговом", "налоговый"),
            ("предпринимателем", "предприниматель"),
            ("лиц", "лицо"),
        ],
    )
    def test_russian_words_map_to_dictionary_lemma(self, word: str, lemma: str) -> None:
        assert normalize_token(word) == lemma

    def test_yo_is_folded_in_lemma(self) -> None:
        assert normalize_token("учёт") == normalize_token("учет") == "учет"

    def test_suffix_stripping_misses_are_matched(self) -> None:
        # The former suffix stemmer kept these apart ("лиц" vs "лицо", "ставк" vs "ставок").
        assert normalize_token("лиц") == normalize_token("лица")
        assert normalize_token("ставок") == normalize_token("ставка")

    def test_english_stemming(self) -> None:
        assert normalize_token("classification") == "classifica"
        assert normalize_token("cat") == "cat"

    def test_unknown_language_passthrough(self) -> None:
        assert normalize_token("12345") == "12345"
