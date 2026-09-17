"""Tests for BM25 ACL pre-filter (Phase D).

Tests that BM25Index stores ACL metadata, pre-filters candidates by
visibility conditions, and maintains backward compatibility with
indexes that have no ACL metadata.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from domain.utils import content_hash
from domain.services.access_control import VisibilityCondition
from domain.value_objects.owner_match import OwnerMatch
from domain.value_objects.visibility import DocumentVisibility
from infrastructure.bm25.bm25_index import BM25Index


# ---------------------------------------------------------------------------
# D1: BM25Index stores ACL metadata
# ---------------------------------------------------------------------------


class TestBM25ACLMetadata:
    def test_add_text_with_acl(self):
        idx = BM25Index(texts=[])
        idx.add_text("public doc", visibility="internal_public", owner_id=None, group_id=None)
        assert idx.doc_visibility == ["internal_public"]
        assert idx.doc_owner_id == [None]
        assert idx.doc_group_id == [None]

    def test_add_text_with_owner(self):
        idx = BM25Index(texts=[])
        idx.add_text("private doc", visibility="internal_private", owner_id=42, group_id=None)
        assert idx.doc_visibility == ["internal_private"]
        assert idx.doc_owner_id == [42]

    def test_add_text_with_group(self):
        idx = BM25Index(texts=[])
        idx.add_text("group doc", visibility="internal_group", owner_id=None, group_id=10)
        assert idx.doc_group_id == [10]

    def test_init_with_acl_lists(self):
        idx = BM25Index(
            texts=["a", "b"],
            doc_visibility=["internal_public", "client_private"],
            doc_owner_id=[None, 100],
            doc_group_id=[None, None],
        )
        assert idx.doc_visibility == ["internal_public", "client_private"]
        assert idx.doc_owner_id == [None, 100]

    def test_default_acl_none(self):
        idx = BM25Index(texts=["hello"])
        assert idx.doc_visibility == [None]
        assert idx.doc_owner_id == [None]
        assert idx.doc_group_id == [None]


# ---------------------------------------------------------------------------
# D2: replace_text and remove_text handle ACL
# ---------------------------------------------------------------------------


class TestBM25ACLReplaceRemove:
    def test_replace_text_updates_acl(self):
        idx = BM25Index(
            texts=["doc1"],
            doc_visibility=["internal_public"],
            doc_owner_id=[None],
            doc_group_id=[None],
        )
        idx.replace_text(0, "doc2", visibility="client_private", owner_id=100)
        assert idx.doc_visibility == ["client_private"]
        assert idx.doc_owner_id == [100]

    def test_remove_text_removes_acl(self):
        idx = BM25Index(
            texts=["doc1", "doc2"],
            doc_visibility=["internal_public", "client_private"],
            doc_owner_id=[None, 100],
            doc_group_id=[None, None],
        )
        idx.remove_text(0)
        assert idx.doc_visibility == ["client_private"]
        assert idx.doc_owner_id == [100]


# ---------------------------------------------------------------------------
# D3: search_with_hashes pre-filters by ACL
# ---------------------------------------------------------------------------


class TestBM25SearchWithACL:
    def _make_index_with_acl(self):
        """Index with mixed visibility docs."""
        idx = BM25Index(
            texts=[
                "public document",  # 0: internal_public
                "private document",  # 1: internal_private, owner=10
                "other private document",  # 2: internal_private, owner=20
                "client document",  # 3: client_private, owner=100
                "group document",  # 4: internal_group, group=5
            ],
            doc_visibility=[
                "internal_public",
                "internal_private",
                "internal_private",
                "client_private",
                "internal_group",
            ],
            doc_owner_id=[None, 10, 20, 100, None],
            doc_group_id=[None, None, None, None, 5],
        )
        return idx

    def test_acl_arguments_are_mandatory_keywords(self):
        idx = self._make_index_with_acl()
        with pytest.raises(TypeError):
            idx.search_with_hashes("document", k=10)
        with pytest.raises(TypeError):
            idx.search_with_hashes("document", 10, [], 10, [])

    @pytest.mark.parametrize("missing", ["visibility_conditions", "user_id", "user_group_ids"])
    @pytest.mark.parametrize("query", ["document", ""])
    def test_none_acl_rejected_before_scoring(self, missing, query):
        idx = self._make_index_with_acl()
        acl = {"visibility_conditions": [], "user_id": 10, "user_group_ids": []}
        acl[missing] = None
        with patch.object(idx, "score") as score, pytest.raises(RuntimeError, match="requires"):
            idx.search_with_hashes(query, **acl)
        score.assert_not_called()

    def test_user_sees_public_and_own_private(self):
        idx = self._make_index_with_acl()
        conditions = [
            VisibilityCondition(visibility=DocumentVisibility.INTERNAL_PUBLIC),
            VisibilityCondition(visibility=DocumentVisibility.INTERNAL_PRIVATE, owner_match=OwnerMatch.SELF),
        ]
        results = idx.search_with_hashes(
            "document",
            k=10,
            visibility_conditions=conditions,
            user_id=10,
            user_group_ids=[],
        )
        hashes = [h for h, _ in results]
        assert content_hash("public document") in hashes
        assert content_hash("private document") in hashes
        assert content_hash("other private document") not in hashes
        assert content_hash("client document") not in hashes

    def test_user_does_not_see_client_private(self):
        idx = self._make_index_with_acl()
        conditions = [
            VisibilityCondition(visibility=DocumentVisibility.INTERNAL_PUBLIC),
            VisibilityCondition(visibility=DocumentVisibility.INTERNAL_PRIVATE, owner_match=OwnerMatch.SELF),
        ]
        results = idx.search_with_hashes(
            "document",
            k=10,
            visibility_conditions=conditions,
            user_id=10,
            user_group_ids=[],
        )
        hashes = [h for h, _ in results]
        assert content_hash("client document") not in hashes

    def test_curator_sees_managed_docs(self):
        idx = self._make_index_with_acl()
        conditions = [
            VisibilityCondition(visibility=DocumentVisibility.INTERNAL_PUBLIC),
            VisibilityCondition(visibility=DocumentVisibility.INTERNAL_PRIVATE, owner_match=OwnerMatch.SELF),
            VisibilityCondition(
                visibility=DocumentVisibility.INTERNAL_PRIVATE,
                owner_match=OwnerMatch.ASSIGNED,
                owner_ids=[20],
            ),
        ]
        results = idx.search_with_hashes(
            "document",
            k=10,
            visibility_conditions=conditions,
            user_id=10,
            user_group_ids=[],
        )
        hashes = [h for h, _ in results]
        assert content_hash("other private document") in hashes

    def test_group_match(self):
        idx = self._make_index_with_acl()
        conditions = [
            VisibilityCondition(visibility=DocumentVisibility.INTERNAL_GROUP, group_match=True),
        ]
        results = idx.search_with_hashes(
            "document",
            k=10,
            visibility_conditions=conditions,
            user_id=99,
            user_group_ids=[5],
        )
        hashes = [h for h, _ in results]
        assert content_hash("group document") in hashes

    def test_group_match_wrong_group(self):
        idx = self._make_index_with_acl()
        conditions = [
            VisibilityCondition(visibility=DocumentVisibility.INTERNAL_GROUP, group_match=True),
        ]
        results = idx.search_with_hashes(
            "document",
            k=10,
            visibility_conditions=conditions,
            user_id=99,
            user_group_ids=[999],
        )
        hashes = [h for h, _ in results]
        assert content_hash("group document") not in hashes

    def test_empty_conditions_no_match(self):
        idx = self._make_index_with_acl()
        results = idx.search_with_hashes(
            "document",
            k=10,
            visibility_conditions=[],
            user_id=10,
            user_group_ids=[],
        )
        assert len(results) == 0


# ---------------------------------------------------------------------------
# D5: Backward compatibility — old indexes without ACL
# ---------------------------------------------------------------------------


class TestBM25ACLBackwardCompat:
    @pytest.mark.parametrize("empty_conditions", [False, True])
    def test_old_index_no_acl_fails_closed(self, empty_conditions):
        idx = BM25Index.from_dict({"texts": ["public doc", "private doc"]})
        assert idx.doc_visibility == [None, None]
        conditions = [
            VisibilityCondition(visibility=DocumentVisibility.INTERNAL_PRIVATE, owner_match=OwnerMatch.SELF),
        ]
        results = idx.search_with_hashes(
            "doc",
            k=10,
            visibility_conditions=[] if empty_conditions else conditions,
            user_id=999,
            user_group_ids=[],
        )
        assert results == []
        assert idx.last_survival_ratio == 0.0
        assert len(idx.search("doc", k=10)) == 2

    def test_to_dict_with_acl(self):
        idx = BM25Index(
            texts=["a"],
            doc_visibility=["internal_public"],
            doc_owner_id=[None],
            doc_group_id=[None],
        )
        d = idx.to_dict()
        assert "doc_visibility" in d
        assert "doc_owner_id" in d
        assert "doc_group_id" in d

    def test_to_dict_without_acl(self):
        idx = BM25Index(texts=["a"])
        d = idx.to_dict()
        assert "doc_visibility" not in d

    def test_from_dict_with_acl(self):
        d = {
            "texts": ["a"],
            "doc_visibility": ["internal_public"],
            "doc_owner_id": [None],
            "doc_group_id": [None],
        }
        idx = BM25Index.from_dict(d)
        assert idx.doc_visibility == ["internal_public"]

    def test_from_dict_without_acl(self):
        d = {"texts": ["a"]}
        idx = BM25Index.from_dict(d)
        assert idx.doc_visibility == [None]

    def test_serialization_roundtrip_with_acl(self):
        idx = BM25Index(
            texts=["hello world", "foo bar"],
            doc_visibility=["internal_public", "client_private"],
            doc_owner_id=[None, 100],
            doc_group_id=[None, None],
        )
        d = idx.to_dict()
        idx2 = BM25Index.from_dict(d)
        assert idx2.doc_visibility == ["internal_public", "client_private"]
        assert idx2.doc_owner_id == [None, 100]
        results = idx2.search_with_hashes(
            "hello",
            k=5,
            visibility_conditions=[VisibilityCondition(visibility=DocumentVisibility.INTERNAL_PUBLIC)],
            user_id=10,
            user_group_ids=[],
        )
        assert len(results) > 0


# ---------------------------------------------------------------------------
# D9: curator_scope_max_ids config
# ---------------------------------------------------------------------------


class TestCuratorScopeMaxIds:
    def test_config_default(self):
        from config import settings

        assert settings.curator_scope_max_ids == 1000


# ---------------------------------------------------------------------------
# D10: Property — _doc_matches_acl ⇔ is_in_search_scope
# ---------------------------------------------------------------------------


class TestBM25PredicateInvariant:
    """Property-based invariant: _doc_matches_acl must agree with is_in_search_scope.

    For every combination of (role x visibility x owner x group), the BM25
    pre-filter predicate must produce the same boolean as the domain's
    is_in_search_scope.  This is the third angle of the invariant triangle
    (domain rule, Qdrant filter, BM25 predicate).
    """

    def _build_test_docs(self):
        """Return list of (label, vis, owner_id, group_id) tuples."""
        return [
            ("pub", "internal_public", None, None),
            ("own_private", "internal_private", 10, None),
            ("other_private", "internal_private", 20, None),
            ("client", "client_private", 100, None),
            ("grp", "internal_group", None, 5),
            ("grp_other", "internal_group", None, 999),
        ]

    def _bm25_matches(self, idx, doc_pos, conditions, user_id, group_ids):
        """Run BM25 _doc_matches_acl and return True/False."""
        return idx._doc_matches_acl(doc_pos, conditions, user_id, group_ids)

    def _run_invariant(
        self,
        user_id,
        group_ids,
        user_role,
        get_conditions_fn,
        managed_client_ids=None,
        managed_internal_ids=None,
        managed_group_ids=None,
    ):
        """Check BM25 predicate matches domain is_in_search_scope for all docs."""
        from types import SimpleNamespace

        labels, vis_list, owner_list, group_list = zip(*self._build_test_docs(), strict=True)
        idx = BM25Index(
            texts=["x"] * len(labels),
            doc_visibility=list(vis_list),
            doc_owner_id=list(owner_list),
            doc_group_id=list(group_list),
        )

        conditions = get_conditions_fn()
        ctx = SimpleNamespace(
            user_id=user_id,
            user_kind="internal",
            user_role=user_role,
            group_ids=group_ids,
            managed_client_ids=managed_client_ids or [],
            managed_internal_ids=managed_internal_ids or [],
            managed_group_ids=managed_group_ids or [],
        )

        for i, (label, vis, owner, grp) in enumerate(self._build_test_docs()):
            doc = SimpleNamespace(visibility=vis, owner_id=owner, group_id=grp)

            from domain.services.access_control import is_in_search_scope

            domain_result = is_in_search_scope(doc, ctx)
            bm25_result = self._bm25_matches(idx, i, conditions, user_id, group_ids)

            assert domain_result == bm25_result, (
                f"Mismatch for {label} ({vis}, owner={owner}, group={grp}): "
                f"is_in_search_scope={domain_result}, _doc_matches_acl={bm25_result}"
            )

    def test_user_invariant(self):
        """USER: BM25 predicate matches domain rule."""
        from domain.services.access_control import get_visibility_conditions
        from domain.value_objects.roles import UserKind, UserRole

        self._run_invariant(
            user_id=10,
            group_ids=[5],
            user_role="user",
            get_conditions_fn=lambda: get_visibility_conditions(
                UserKind.INTERNAL,
                10,
                [5],
                for_list=False,
                user_role=UserRole.USER,
            ),
        )

    def test_admin_invariant(self):
        """ADMIN: BM25 predicate matches domain rule (no CLIENT_PRIVATE bonus)."""
        from domain.services.access_control import get_visibility_conditions
        from domain.value_objects.roles import UserKind, UserRole

        self._run_invariant(
            user_id=1,
            group_ids=[],
            user_role="admin",
            get_conditions_fn=lambda: get_visibility_conditions(
                UserKind.INTERNAL,
                1,
                [],
                for_list=False,
                user_role=UserRole.ADMIN,
            ),
        )

    def test_curator_with_managed_invariant(self):
        """CURATOR with managed_ids: BM25 predicate matches domain rule."""
        from domain.services.access_control import get_visibility_conditions
        from domain.value_objects.roles import UserKind, UserRole

        self._run_invariant(
            user_id=10,
            group_ids=[5],
            user_role="curator",
            get_conditions_fn=lambda: get_visibility_conditions(
                UserKind.INTERNAL,
                10,
                [5],
                for_list=False,
                user_role=UserRole.CURATOR,
                managed_internal_ids=[20],
                managed_client_ids=[100],
            ),
            managed_internal_ids=[20],
            managed_client_ids=[100],
        )

    def test_acl_none_in_index_denied(self):
        idx = BM25Index(texts=["pub", "priv"], doc_visibility=[None, None])
        from domain.services.access_control import VisibilityCondition
        from domain.value_objects.owner_match import OwnerMatch
        from domain.value_objects.visibility import DocumentVisibility

        conditions = [
            VisibilityCondition(visibility=DocumentVisibility.INTERNAL_PRIVATE, owner_match=OwnerMatch.SELF),
        ]
        assert not idx._doc_matches_acl(0, conditions, user_id=1, user_group_ids=[])
        assert not idx._doc_matches_acl(1, conditions, user_id=1, user_group_ids=[])
