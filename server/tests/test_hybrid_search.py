"""Tests for BM25, tokenizer, RRF merge, persistence.

All pure functions, no Qdrant/Ollama needed.
"""

import sys
import json
from unittest.mock import AsyncMock

import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from domain.shared.hashing import content_hash
from domain.knowledge.enums import DocumentVisibility
from domain.rag.retrieval import rrf_merge  # noqa: E402
from infrastructure.bm25.bm25_index import BM25Index  # noqa: E402
from infrastructure.bm25.persistence import BM25_S3_KEY, load_bm25_index_from_s3, save_bm25_index_to_s3  # noqa: E402
from infrastructure.bm25.tokenizer import tokenize, tokenize_raw  # noqa: E402

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_basic_english(self):
        assert tokenize("Hello World") == ["hello", "world"]

    def test_russian_text(self):
        tokens = tokenize_raw("Постановление от 14.04.2026 года")
        assert "постановление" in tokens
        assert "14" in tokens
        assert "04" in tokens
        assert "2026" in tokens
        assert "года" in tokens

    def test_short_tokens_filtered(self):
        # Single char tokens "a", "x" are filtered (< 2 chars)
        tokens = tokenize("a x")
        assert tokens == []

    def test_punctuation_removed(self):
        tokens = tokenize_raw("маркировка: код, номер!")
        assert "маркировка" in tokens
        assert "код" in tokens
        assert "номер" in tokens

    def test_numbers_preserved(self):
        tokens = tokenize("статья 14 пункт 32")
        assert "14" in tokens
        assert "32" in tokens

    def test_empty_string(self):
        assert tokenize("") == []


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
        idx = BM25Index(texts)
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
        idx = BM25Index(texts)
        results = idx.search("14.04.2026", k=3)
        # Exact date should rank first
        assert texts[results[0][0]] == "постановление от 14.04.2026"

    def test_empty_query(self):
        idx = BM25Index(["text one", "text two"])
        assert idx.search("", k=5) == []

    def test_single_doc(self):
        idx = BM25Index(["only document"])
        results = idx.search("only", k=5)
        assert len(results) == 1

    def test_k_larger_than_docs(self):
        idx = BM25Index(["alpha", "bravo"])
        results = idx.search("alpha bravo", k=10)
        assert len(results) == 2

    def test_search_with_hashes(self):
        texts = ["маркировка товаров", "штрафы за нарушение"]
        from domain.knowledge.access_control import VisibilityCondition
        from domain.knowledge.enums import DocumentVisibility

        idx = BM25Index(texts, doc_visibility=["internal_public"] * len(texts))
        results = idx.search_with_hashes(
            "маркировка",
            k=1,
            visibility_conditions=[VisibilityCondition(visibility=DocumentVisibility.INTERNAL_PUBLIC)],
            user_id=10,
            user_group_ids=[],
        )
        assert len(results) == 1
        h, score = results[0]
        assert isinstance(h, str)
        assert len(h) == 16  # content_hash is 16 hex chars
        assert score > 0

    def test_serialization_roundtrip(self):
        texts = ["постановление 14.04.2026", "маркировка кодов", "штрафы"]
        idx = BM25Index(texts)
        data = idx.to_dict()
        idx2 = BM25Index.from_dict(data)
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
        merged = rrf_merge(dense, sparse)
        # "a" and "b" appear in both, should rank highest
        assert merged[0] in ("a", "b")
        assert merged[1] in ("a", "b")
        # "c" only in dense, "d" only in sparse
        assert "c" in merged
        assert "d" in merged

    def test_deduplication(self):
        dense = [("a", 0.9), ("a", 0.8)]  # duplicate
        sparse = [("a", 5.0)]
        merged = rrf_merge(dense, sparse)
        assert merged.count("a") == 1

    def test_empty_dense(self):
        sparse = [("a", 5.0), ("b", 3.0)]
        merged = rrf_merge([], sparse)
        assert merged == ["a", "b"]

    def test_empty_sparse(self):
        dense = [("a", 0.9), ("b", 0.8)]
        merged = rrf_merge(dense, [])
        assert merged == ["a", "b"]

    def test_both_empty(self):
        assert rrf_merge([], []) == []

    def test_weight_affects_ranking(self):
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
    def test_deterministic(self):
        h1 = content_hash("hello world")
        h2 = content_hash("hello world")
        assert h1 == h2

    def test_different_inputs_different_hashes(self):
        h1 = content_hash("hello")
        h2 = content_hash("world")
        assert h1 != h2

    def test_length(self):
        h = content_hash("test")
        assert len(h) == 16


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    @pytest.mark.asyncio
    async def test_s3_roundtrip_preserves_scores_and_acl(self, tmp_path):
        index = BM25Index(
            ["маркировка маркировка", "штрафы", "постановление"],
            doc_visibility=[
                DocumentVisibility.INTERNAL_PUBLIC,
                DocumentVisibility.INTERNAL_PRIVATE,
                DocumentVisibility.INTERNAL_PUBLIC,
            ],
            doc_owner_id=[1, 2, 3],
            doc_group_id=[None, 7, None],
        )
        storage = AsyncMock()
        await save_bm25_index_to_s3(index, storage)
        key, payload = storage.upload_file.call_args.args
        assert key == BM25_S3_KEY
        assert json.loads(payload) == index.to_dict()
        path = tmp_path / "download.json"
        path.write_bytes(payload)
        storage.download_to_temp.return_value = path
        loaded = await load_bm25_index_from_s3(storage)
        assert loaded.to_dict() == index.to_dict()
        assert loaded.search("маркировка") == index.search("маркировка")
        assert not path.exists()


@pytest.mark.parametrize("operation", ["replace", "remove", "remove_last", "add"])
def test_incremental_statistics_match_rebuild_with_repeated_terms(operation):
    index = BM25Index(["маркировка маркировка", "маркировка товары", "штрафы"])

    if operation == "replace":
        index.replace_text(0, "товары товары")
    elif operation == "remove":
        index.remove_text(0)
    elif operation == "remove_last":
        index.remove_text(index.n_docs - 1)
    else:
        index.add_text("товары товары")

    rebuilt = BM25Index.from_dict(index.to_dict())
    assert index.doc_freq == rebuilt.doc_freq
    assert index.token_freqs == rebuilt.token_freqs
    assert index.inverted_index == rebuilt.inverted_index
    assert index.doc_lens == rebuilt.doc_lens
    assert index.avgdl == pytest.approx(rebuilt.avgdl)

    for position, value in enumerate(index.hashes):
        assert index.find_by_hash(value) == position

    assert index.search("маркировка товары") == rebuilt.search("маркировка товары")
