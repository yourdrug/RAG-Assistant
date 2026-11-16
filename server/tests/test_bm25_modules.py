"""Stemmer tests for the BM25 tokenizer used by Qdrant sparse vectors."""

from infrastructure.bm25.stemmer import stem_token


# ---------------------------------------------------------------------------
# Stemmer (unique to this file — not tested in test_hybrid_search.py)
# ---------------------------------------------------------------------------


class TestStemmer:
    def test_russian_stemming(self) -> None:
        stemmed = stem_token("маркировка")
        assert len(stemmed) < len("маркировка")
        assert stemmed == "маркировк"

    def test_english_stemming(self) -> None:
        stemmed = stem_token("classification")
        assert stemmed == "classifica"

    def test_short_word_not_stemmed(self) -> None:
        assert stem_token("мир") == "мир"
        assert stem_token("cat") == "cat"

    def test_unknown_language_passthrough(self) -> None:
        assert stem_token("12345") == "12345"
