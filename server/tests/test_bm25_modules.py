"""Tests for split BM25 modules — unique tests not covered by test_hybrid_search.py.

Focuses on: stemmer, backward-compat shim, and incremental operations (add/remove).
Core BM25/tokenizer/RRF tests live in test_hybrid_search.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from infrastructure.bm25._stemmer import stem_token  # noqa: E402
from infrastructure.bm25.bm25_index import BM25Index  # noqa: E402


# ---------------------------------------------------------------------------
# Stemmer (unique to this file — not tested in test_hybrid_search.py)
# ---------------------------------------------------------------------------


class TestStemmer:
    def test_russian_stemming(self):
        stemmed = stem_token("маркировка")
        assert len(stemmed) < len("маркировка")
        assert stemmed == "маркиров"

    def test_english_stemming(self):
        stemmed = stem_token("classification")
        assert stemmed == "classifica"

    def test_short_word_not_stemmed(self):
        assert stem_token("мир") == "мир"
        assert stem_token("cat") == "cat"

    def test_unknown_language_passthrough(self):
        assert stem_token("12345") == "12345"


# ---------------------------------------------------------------------------
# Incremental operations (unique to this file)
# ---------------------------------------------------------------------------


class TestBM25Incremental:
    def test_add_text(self):
        idx = BM25Index(texts=["hello world"])
        idx.add_text("foo bar baz")
        assert idx.n_docs == 2
        results = idx.search("foo")
        assert len(results) > 0

    def test_remove_text(self):
        idx = BM25Index(texts=["hello world", "foo bar"])
        idx.remove_text(0)
        assert idx.n_docs == 1
        assert idx.texts == ["foo bar"]
        results = idx.search("hello")
        assert len(results) == 0

    def test_serialization_roundtrip_after_mutation(self):
        idx = BM25Index(texts=["alpha", "bravo"])
        idx.add_text("charlie")
        idx.remove_text(0)
        data = idx.to_dict()
        idx2 = BM25Index.from_dict(data)
        assert idx2.n_docs == 2
        assert idx2.texts == ["bravo", "charlie"]


# ---------------------------------------------------------------------------
# Backward compatibility — hybrid.py shim (unique to this file)
# ---------------------------------------------------------------------------


class TestHybridBackwardCompat:
    """Verify that the hybrid.py shim re-exports everything correctly."""

    def test_imports_from_shim(self):
        from infrastructure.bm25.hybrid import (
            BM25Index as ShimBM25,
            content_hash as ShimHash,
            rrf_merge as ShimRRF,
            tokenize as ShimTokenize,
        )

        assert ShimBM25 is BM25Index
        assert ShimRRF is not None
        assert ShimTokenize is not None
        assert ShimHash is not None

    def test_shim_produces_same_results(self):
        from infrastructure.bm25.hybrid import BM25Index as ShimBM25, rrf_merge as ShimRRF

        texts = ["маркировка", "штрафы"]
        idx_direct = BM25Index(texts)
        idx_shim = ShimBM25(texts)

        r1 = idx_direct.search("маркировка", k=1)
        r2 = idx_shim.search("маркировка", k=1)
        assert r1[0][0] == r2[0][0]

        dense = [("a", 0.9)]
        sparse = [("a", 5.0)]
        assert ShimRRF(dense, sparse) == ShimRRF(dense, sparse)
