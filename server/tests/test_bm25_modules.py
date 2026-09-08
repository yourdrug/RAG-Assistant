"""Tests for split BM25 modules — direct imports from the new submodules.

Verifies that the split modules work independently and produce the same
results as the old monolithic hybrid.py.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from infrastructure.bm25._stemmer import stem_token  # noqa: E402
from infrastructure.bm25._tokenizer import tokenize, tokenize_raw  # noqa: E402
from infrastructure.bm25.bm25_index import BM25Index  # noqa: E402
from infrastructure.bm25.persistence import (  # noqa: E402
    load_bm25_index,
    save_bm25_index,
)
from infrastructure.bm25.rrf import rrf_merge  # noqa: E402
from domain.utils import content_hash  # noqa: E402


# ---------------------------------------------------------------------------
# Stemmer
# ---------------------------------------------------------------------------


class TestStemmer:
    def test_russian_stemming(self):
        # "маркировка" should stem to something shorter
        stemmed = stem_token("маркировка")
        assert len(stemmed) < len("маркировка")
        assert stemmed == "маркиров"  # strip "ка" suffix

    def test_english_stemming(self):
        stemmed = stem_token("classification")
        assert stemmed == "classifica"  # strip "tion" suffix

    def test_short_word_not_stemmed(self):
        # Words under 5 chars are not stemmed
        assert stem_token("мир") == "мир"
        assert stem_token("cat") == "cat"

    def test_unknown_language(self):
        # Non-Russian, non-English tokens pass through
        assert stem_token("12345") == "12345"


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


class TestTokenizer:
    def test_basic_english(self):
        assert tokenize("Hello World") == ["hello", "world"]

    def test_russian_text(self):
        tokens = tokenize("постановление от 14.04.2026 года")
        # "постановление" stems to "постановл" (strip "ение")
        assert "постановл" in tokens

    def test_short_tokens_filtered(self):
        assert tokenize("a x") == []

    def test_numbers_preserved(self):
        tokens = tokenize("статья 14 пункт 32")
        assert "14" in tokens
        assert "32" in tokens

    def test_empty_string(self):
        assert tokenize("") == []

    def test_tokenize_raw_no_stemming(self):
        tokens = tokenize_raw("маркировка товаров")
        assert "маркировка" in tokens
        # Raw should NOT be stemmed
        assert "маркировк" not in tokens


# ---------------------------------------------------------------------------
# BM25Index — direct import
# ---------------------------------------------------------------------------


class TestBM25IndexDirect:
    def test_basic_search(self):
        texts = [
            "код маркировки товаров",
            "штрафы за нарушение маркировки",
            "порядок получения кода",
        ]
        idx = BM25Index(texts)
        results = idx.search("маркировка", k=2)
        assert len(results) == 2
        result_texts = {texts[i] for i, _ in results}
        assert "код маркировки товаров" in result_texts

    def test_empty_query(self):
        idx = BM25Index(["text one", "text two"])
        assert idx.search("", k=5) == []

    def test_search_with_hashes(self):
        texts = ["маркировка товаров", "штрафы за нарушение"]
        idx = BM25Index(texts)
        results = idx.search_with_hashes("маркировка", k=1)
        assert len(results) == 1
        h, score = results[0]
        assert isinstance(h, str)
        assert len(h) == 16

    def test_serialization_roundtrip(self):
        texts = ["постановление 14.04.2026", "маркировка кодов", "штрафы"]
        idx = BM25Index(texts)
        data = idx.to_dict()
        idx2 = BM25Index.from_dict(data)
        assert idx2.texts == texts
        assert idx2.n_docs == 3
        r1 = idx.search("маркировка", k=2)
        r2 = idx2.search("маркировка", k=2)
        assert [i for i, _ in r1] == [i for i, _ in r2]

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


# ---------------------------------------------------------------------------
# RRF — direct import
# ---------------------------------------------------------------------------


class TestRRFDirect:
    def test_basic_merge(self):
        dense = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
        sparse = [("b", 5.0), ("a", 3.0), ("d", 2.0)]
        merged = rrf_merge(dense, sparse)
        assert merged[0] in ("a", "b")
        assert "c" in merged
        assert "d" in merged

    def test_empty_inputs(self):
        assert rrf_merge([], []) == []
        assert rrf_merge([], [("a", 1.0)]) == ["a"]
        assert rrf_merge([("a", 1.0)], []) == ["a"]

    def test_deduplication(self):
        dense = [("a", 0.9), ("a", 0.8)]
        sparse = [("a", 5.0)]
        merged = rrf_merge(dense, sparse)
        assert merged.count("a") == 1

    def test_weight_effect(self):
        dense = [("a", 0.9), ("b", 0.8)]
        sparse = [("b", 5.0), ("a", 3.0)]
        merged_default = rrf_merge(dense, sparse)
        merged_sparse = rrf_merge(dense, sparse, sparse_weight=10.0)
        assert merged_sparse.index("b") <= merged_default.index("b")


# ---------------------------------------------------------------------------
# Persistence — direct import
# ---------------------------------------------------------------------------


class TestPersistenceDirect:
    def test_save_and_load(self):
        texts = ["маркировка", "штрафы", "постановление"]
        idx = BM25Index(texts)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bm25.json"
            save_bm25_index(idx, path)
            loaded = load_bm25_index(path)
            assert loaded is not None
            assert loaded.n_docs == 3

    def test_load_nonexistent(self):
        assert load_bm25_index(Path("/tmp/nonexistent_bm25_test.json")) is None


# ---------------------------------------------------------------------------
# Backward compatibility — hybrid.py shim
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
        assert ShimHash is content_hash
        assert ShimRRF is rrf_merge
        assert ShimTokenize is tokenize

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
        assert rrf_merge(dense, sparse) == ShimRRF(dense, sparse)
