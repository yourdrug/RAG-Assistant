"""Tests for the sparse tokenizer, RRF merge and content hashes.

All pure functions, no Qdrant/LLM needed.
"""

from domain.rag.ranking import rrf_merge
from domain.shared.hashing import content_hash
from infrastructure.bm25.tokenizer import tokenize, tokenize_raw


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_basic_english(self) -> None:
        assert tokenize("Hello World") == ["hello", "world"]

    def test_russian_text(self) -> None:
        tokens = tokenize_raw("Постановление от 14.04.2026 года")
        assert "постановление" in tokens
        assert "14" in tokens
        assert "04" in tokens
        assert "2026" in tokens
        assert "года" in tokens

    def test_short_tokens_filtered(self) -> None:
        # Single char tokens "a", "x" are filtered (< 2 chars)
        tokens = tokenize("a x")
        assert tokens == []

    def test_punctuation_removed(self) -> None:
        tokens = tokenize_raw("маркировка: код, номер!")
        assert "маркировка" in tokens
        assert "код" in tokens
        assert "номер" in tokens

    def test_numbers_preserved(self) -> None:
        tokens = tokenize("статья 14 пункт 32")
        assert "14" in tokens
        assert "32" in tokens

    def test_empty_string(self) -> None:
        assert tokenize("") == []


# ---------------------------------------------------------------------------
# RRF Merge
# ---------------------------------------------------------------------------


class TestRRFMerge:
    def test_basic_merge(self) -> None:
        dense = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
        sparse = [("b", 5.0), ("a", 3.0), ("d", 2.0)]
        merged = rrf_merge(dense, sparse)
        # "a" and "b" appear in both, should rank highest
        assert merged[0] in ("a", "b")
        assert merged[1] in ("a", "b")
        # "c" only in dense, "d" only in sparse
        assert "c" in merged
        assert "d" in merged

    def test_deduplication(self) -> None:
        dense = [("a", 0.9), ("a", 0.8)]  # duplicate
        sparse = [("a", 5.0)]
        merged = rrf_merge(dense, sparse)
        assert merged.count("a") == 1

    def test_empty_dense(self) -> None:
        sparse = [("a", 5.0), ("b", 3.0)]
        merged = rrf_merge([], sparse)
        assert merged == ["a", "b"]

    def test_empty_sparse(self) -> None:
        dense = [("a", 0.9), ("b", 0.8)]
        merged = rrf_merge(dense, [])
        assert merged == ["a", "b"]

    def test_both_empty(self) -> None:
        assert rrf_merge([], []) == []

    def test_weight_affects_ranking(self) -> None:
        dense = [("a", 0.9), ("b", 0.8)]
        sparse = [("b", 5.0), ("a", 3.0)]
        # Default weights: equal
        merged_default = rrf_merge(dense, sparse)
        # High sparse weight: sparse results dominate
        merged_sparse_heavy = rrf_merge(dense, sparse, sparse_weight=10.0)
        # "b" is #1 in sparse, should rank higher with heavy sparse weight
        assert merged_sparse_heavy.index("b") <= merged_default.index("b")


# ---------------------------------------------------------------------------
# Content Hash
# ---------------------------------------------------------------------------


class TestContentHash:
    def test_deterministic(self) -> None:
        h1 = content_hash("hello world")
        h2 = content_hash("hello world")
        assert h1 == h2

    def test_different_inputs_different_hashes(self) -> None:
        h1 = content_hash("hello")
        h2 = content_hash("world")
        assert h1 != h2

    def test_length(self) -> None:
        h = content_hash("test")
        assert len(h) == 16
