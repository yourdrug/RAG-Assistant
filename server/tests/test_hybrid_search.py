"""
Tests for infrastructure/ml/hybrid.py — BM25, tokenizer, RRF merge, persistence.
All pure functions, no Qdrant/Ollama needed.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import infrastructure.ml.hybrid as hybrid  # noqa: E402

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_basic_english(self):
        assert hybrid.tokenize("Hello World") == ["hello", "world"]

    def test_russian_text(self):
        tokens = hybrid.tokenize("Постановление от 14.04.2026 года")
        assert "постановление" in tokens
        assert "14" in tokens
        assert "04" in tokens
        assert "2026" in tokens
        assert "года" in tokens

    def test_short_tokens_filtered(self):
        # Single char tokens "a", "x" are filtered (< 2 chars)
        tokens = hybrid.tokenize("a x")
        assert tokens == []

    def test_punctuation_removed(self):
        tokens = hybrid.tokenize("маркировка: код, номер!")
        assert "маркировка" in tokens
        assert "код" in tokens
        assert "номер" in tokens

    def test_numbers_preserved(self):
        tokens = hybrid.tokenize("статья 14 пункт 32")
        assert "14" in tokens
        assert "32" in tokens

    def test_empty_string(self):
        assert hybrid.tokenize("") == []


# ---------------------------------------------------------------------------
# BM25 Index
# ---------------------------------------------------------------------------


class TestBM25Index:
    def test_basic_search(self):
        texts = [
            "код маркировки товаров",
            "штрафы за нарушение маркировки",
            "порядок получения кода",
        ]
        idx = hybrid.BM25Index(texts)
        results = idx.search("маркировка", k=2)
        assert len(results) == 2
        # Both docs about маркировка should rank higher
        result_texts = {texts[i] for i, _ in results}
        assert "код маркировки товаров" in result_texts
        assert "штрафы за нарушение маркировки" in result_texts

    def test_exact_number_match(self):
        texts = [
            "постановление от 14.04.2026",
            "постановление от 01.01.2020",
            "статья 14 пункт 3",
        ]
        idx = hybrid.BM25Index(texts)
        results = idx.search("14.04.2026", k=3)
        # Exact date should rank first
        assert texts[results[0][0]] == "постановление от 14.04.2026"

    def test_empty_query(self):
        idx = hybrid.BM25Index(["text one", "text two"])
        assert idx.search("", k=5) == []

    def test_single_doc(self):
        idx = hybrid.BM25Index(["only document"])
        results = idx.search("only", k=5)
        assert len(results) == 1

    def test_k_larger_than_docs(self):
        idx = hybrid.BM25Index(["alpha", "bravo"])
        results = idx.search("alpha bravo", k=10)
        assert len(results) == 2

    def test_search_with_hashes(self):
        texts = ["маркировка товаров", "штрафы за нарушение"]
        idx = hybrid.BM25Index(texts)
        results = idx.search_with_hashes("маркировка", k=1)
        assert len(results) == 1
        h, score = results[0]
        assert isinstance(h, str)
        assert len(h) == 16  # content_hash is 16 hex chars
        assert score > 0

    def test_serialization_roundtrip(self):
        texts = ["постановление 14.04.2026", "маркировка кодов", "штрафы"]
        idx = hybrid.BM25Index(texts)
        data = idx.to_dict()
        idx2 = hybrid.BM25Index.from_dict(data)
        assert idx2.texts == texts
        assert idx2.n_docs == 3
        # Search results should be identical
        r1 = idx.search("маркировка", k=2)
        r2 = idx2.search("маркировка", k=2)
        assert [i for i, _ in r1] == [i for i, _ in r2]


# ---------------------------------------------------------------------------
# RRF Merge
# ---------------------------------------------------------------------------


class TestRRFMerge:
    def test_basic_merge(self):
        dense = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
        sparse = [("b", 5.0), ("a", 3.0), ("d", 2.0)]
        merged = hybrid.rrf_merge(dense, sparse)
        # "a" and "b" appear in both, should rank highest
        assert merged[0] in ("a", "b")
        assert merged[1] in ("a", "b")
        # "c" only in dense, "d" only in sparse
        assert "c" in merged
        assert "d" in merged

    def test_deduplication(self):
        dense = [("a", 0.9), ("a", 0.8)]  # duplicate
        sparse = [("a", 5.0)]
        merged = hybrid.rrf_merge(dense, sparse)
        assert merged.count("a") == 1

    def test_empty_dense(self):
        sparse = [("a", 5.0), ("b", 3.0)]
        merged = hybrid.rrf_merge([], sparse)
        assert merged == ["a", "b"]

    def test_empty_sparse(self):
        dense = [("a", 0.9), ("b", 0.8)]
        merged = hybrid.rrf_merge(dense, [])
        assert merged == ["a", "b"]

    def test_both_empty(self):
        assert hybrid.rrf_merge([], []) == []

    def test_weight_affects_ranking(self):
        dense = [("a", 0.9), ("b", 0.8)]
        sparse = [("b", 5.0), ("a", 3.0)]
        # Default weights: equal
        merged_default = hybrid.rrf_merge(dense, sparse)
        # High sparse weight: sparse results dominate
        merged_sparse_heavy = hybrid.rrf_merge(dense, sparse, sparse_weight=10.0)
        # "b" is #1 in sparse, should rank higher with heavy sparse weight
        assert merged_sparse_heavy.index("b") <= merged_default.index("b")


# ---------------------------------------------------------------------------
# Content Hash
# ---------------------------------------------------------------------------


class TestContentHash:
    def test_deterministic(self):
        h1 = hybrid.content_hash("hello world")
        h2 = hybrid.content_hash("hello world")
        assert h1 == h2

    def test_different_inputs_different_hashes(self):
        h1 = hybrid.content_hash("hello")
        h2 = hybrid.content_hash("world")
        assert h1 != h2

    def test_length(self):
        h = hybrid.content_hash("test")
        assert len(h) == 16


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_and_load_bm25(self):
        texts = ["маркировка", "штрафы", "постановление"]
        idx = hybrid.BM25Index(texts)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bm25.json"
            hybrid.save_bm25_index(idx, path)
            loaded = hybrid.load_bm25_index(path)
            assert loaded is not None
            assert loaded.n_docs == 3
            r1 = idx.search("маркировка", k=1)
            r2 = loaded.search("маркировка", k=1)
            assert r1[0][0] == r2[0][0]

    def test_load_nonexistent_returns_none(self):
        assert hybrid.load_bm25_index(Path("/tmp/nonexistent_bm25.json")) is None

    def test_save_creates_parent_dirs(self):
        idx = hybrid.BM25Index(["test"])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "bm25.json"
            hybrid.save_bm25_index(idx, path)
            assert path.exists()
