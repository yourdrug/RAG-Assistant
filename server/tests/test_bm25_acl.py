"""Characterization tests for BM25 ACL behavior.

These tests document the CURRENT (pre-fix) behavior:
- BM25 search_with_hashes returns ONLY (content_hash, score) — no page_content/metadata
- BM25 index is global (all users' chunks) — no ACL filtering
- Sparse candidates survive ACL only through resolve_hashes_batch post-filter
- sparse_survival_ratio = len(resolved) / len(sparse_results)

All tests MUST pass BEFORE any Phase B-D changes. They serve as regression baseline.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from domain.utils import content_hash
from infrastructure.bm25.bm25_index import BM25Index


# ---------------------------------------------------------------------------
# A1: BM25 search_with_hashes returns only hashes, not content
# ---------------------------------------------------------------------------


class TestBM25ReturnsOnlyHashes:
    """Verify that search_with_hashes exposes NO content/metadata.

    Security invariant: BM25 index NEVER returns page_content or payload.
    Only (content_hash, score) pairs are exposed. Content is delivered
    exclusively through Qdrant under ACL filter via resolve_hashes_batch.
    """

    def test_search_with_hashes_returns_tuple_of_str_float(self):
        idx = BM25Index(texts=["hello world", "foo bar", "test document"])
        results = idx.search_with_hashes("hello", k=5)
        assert len(results) > 0
        for item in results:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], str)  # content_hash
            assert isinstance(item[1], float)  # score

    def test_search_with_hashes_does_not_contain_page_content(self):
        texts = ["secret document about project X", "public meeting notes"]
        idx = BM25Index(texts=texts)
        results = idx.search_with_hashes("document", k=10)
        for h, _score in results:
            # Hash must not contain the original text
            assert "secret" not in h.lower()
            assert "meeting" not in h.lower()
            # Hash must be a hex string (content_hash output)
            assert all(c in "0123456789abcdef" for c in h)

    def test_search_with_hashes_returns_only_hashes_in_results(self):
        idx = BM25Index(texts=["alpha", "beta", "gamma"])
        results = idx.search_with_hashes("alpha", k=10)
        assert len(results) >= 1
        # The hash of "alpha" should be in results
        expected_hash = content_hash("alpha")
        returned_hashes = [h for h, _ in results]
        assert expected_hash in returned_hashes

    def test_search_with_hashes_score_is_bm25_score(self):
        idx = BM25Index(texts=["the quick brown fox", "the lazy dog"])
        results = idx.search_with_hashes("quick fox", k=10)
        # Scores should be positive and ordered descending
        scores = [s for _, s in results]
        assert all(s > 0 for s in scores)
        assert scores == sorted(scores, reverse=True)

    def test_bm25_index_stores_texts_in_memory(self):
        """Document that texts are in process memory (known limitation)."""
        secret = "classified document content"
        idx = BM25Index(texts=[secret])
        # Texts are stored in memory — this is a known limitation
        assert secret in idx.texts


# ---------------------------------------------------------------------------
# A1b: BM25 index is global — no ACL awareness
# ---------------------------------------------------------------------------


class TestBM25GlobalIndex:
    """BM25Index contains ALL users' chunks — no visibility filtering.

    Today the index has zero awareness of ACL. Sparse search returns
    candidates from ALL users, which are then post-filtered by
    resolve_hashes_batch via Qdrant scroll with access_filter.
    """

    def test_index_contains_all_texts_regardless_of_owner(self):
        # Simulate chunks from different "owners"
        idx = BM25Index(texts=[
            "owner1 private document",
            "owner2 private document",
            "public document",
        ])
        # Search finds ALL matching texts — no owner filtering
        results = idx.search_with_hashes("document", k=10)
        hashes = [h for h, _ in results]
        assert content_hash("owner1 private document") in hashes
        assert content_hash("owner2 private document") in hashes
        assert content_hash("public document") in hashes

    def test_search_with_hashes_has_no_acl_parameter(self):
        """Document: search_with_hashes has no visibility_conditions param."""
        import inspect
        sig = inspect.signature(BM25Index.search_with_hashes)
        # Currently only (query, k) — no ACL parameter
        params = list(sig.parameters.keys())
        assert "visibility_conditions" not in params
        assert "access_filter" not in params


# ---------------------------------------------------------------------------
# A2: Sparse survival ratio concept
# ---------------------------------------------------------------------------


class TestSparseSurvivalRatio:
    """Characterize the sparse_survival_ratio metric.

    After resolve_hashes_batch with access_filter, only a fraction of
    BM25 candidates survive (those whose owner matches the ACL).
    sparse_survival_ratio = len(resolved) / len(sparse_results)

    Low ratio → BM25 is wasteful (returns many candidates that fail ACL)
    High ratio → BM25 is efficient for this user's scope
    """

    def test_sparse_survival_ratio_concept(self):
        """Demonstrate the ratio calculation."""
        # Simulate: BM25 returns 10 hashes, but only 3 pass ACL
        sparse_results = [
            (content_hash(f"doc {i}"), 1.0 - i * 0.1)
            for i in range(10)
        ]
        # After resolve_hashes_batch, only indices 0,1,2 survived ACL
        resolved_hashes = {content_hash(f"doc {i}") for i in range(3)}
        resolved_count = sum(1 for h, _ in sparse_results if h in resolved_hashes)
        ratio = resolved_count / len(sparse_results) if sparse_results else 0.0
        assert ratio == 0.3  # 30% survival

    def test_sparse_survival_ratio_zero_when_no_sparse(self):
        """No sparse results → ratio is 0 (not division by zero)."""
        sparse_results = []
        resolved_count = 0
        ratio = resolved_count / len(sparse_results) if sparse_results else 0.0
        assert ratio == 0.0

    def test_sparse_survival_ratio_one_when_all_survive(self):
        """All sparse results pass ACL → ratio is 1.0."""
        sparse_results = [("h1", 1.0), ("h2", 0.9), ("h3", 0.8)]
        resolved_hashes = {"h1", "h2", "h3"}
        resolved_count = sum(1 for h, _ in sparse_results if h in resolved_hashes)
        ratio = resolved_count / len(sparse_results)
        assert ratio == 1.0

    def test_user_with_narrow_scope_has_low_survival(self):
        """User seeing only own docs: most BM25 candidates are from others → low ratio."""
        # BM25 global index has 100 docs, user owns 5
        all_hashes = [content_hash(f"other_user_doc_{i}") for i in range(95)]
        own_hashes = [content_hash(f"my_doc_{i}") for i in range(5)]
        bm25_results = [(h, 1.0) for h in (all_hashes[:9] + own_hashes[:1])]
        # Only 1 of 10 is from this user
        resolved_hashes = set(own_hashes)
        resolved_count = sum(1 for h, _ in bm25_results if h in resolved_hashes)
        ratio = resolved_count / len(bm25_results)
        assert ratio == 0.1  # 10% — most BM25 results wasted

    def test_admin_with_wide_scope_has_high_survival(self):
        """Admin sees all docs: most BM25 candidates survive → high ratio."""
        all_hashes = [content_hash(f"doc_{i}") for i in range(10)]
        bm25_results = [(h, 1.0) for h in all_hashes]
        # Admin can see everything
        resolved_hashes = set(all_hashes)
        resolved_count = sum(1 for h, _ in bm25_results if h in resolved_hashes)
        ratio = resolved_count / len(bm25_results)
        assert ratio == 1.0  # 100% — no waste
