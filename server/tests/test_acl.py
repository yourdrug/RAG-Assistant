"""Tests for access control logic.

Domain rules: domain/services/access_control.py
Qdrant filter: infrastructure/acl.py
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import pytest  # noqa: E402
from domain.exceptions import ValidationError  # noqa: E402
from domain.services import (  # noqa: E402
    can_view_document,
    compute_owner_and_group,
    validate_document_visibility,
)
from domain.services.access_control import ALLOWED_VISIBILITY_FOR_KIND, is_in_search_scope  # noqa: E402
from domain.services import get_visibility_conditions  # noqa: E402
from domain.value_objects.roles import UserKind, UserRole
from domain.value_objects.user_context import UserContext
from domain.value_objects.visibility import DocumentVisibility
from infrastructure.repositories.vector.acl import build_qdrant_filter  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _internal_user(user_id=1, role=UserRole.USER, group_ids=None):
    return UserContext(
        user_id=user_id,
        user_kind=UserKind.INTERNAL,
        user_role=role,
        group_ids=group_ids or [],
    )


def _client_user(user_id=100, group_ids=None):
    return UserContext(
        user_id=user_id,
        user_kind=UserKind.CLIENT,
        user_role=UserRole.USER,
        group_ids=group_ids or [],
    )


# ---------------------------------------------------------------------------
# ALLOWED_VISIBILITY_FOR_KIND
# ---------------------------------------------------------------------------


class TestAllowedVisibility:
    def test_internal_allowed_visibilities(self):
        assert ALLOWED_VISIBILITY_FOR_KIND[UserKind.INTERNAL] == {
            DocumentVisibility.INTERNAL_PUBLIC,
            DocumentVisibility.INTERNAL_GROUP,
            DocumentVisibility.INTERNAL_PRIVATE,
            DocumentVisibility.CLIENT_PRIVATE,
        }

    def test_client_allowed_visibilities(self):
        assert ALLOWED_VISIBILITY_FOR_KIND[UserKind.CLIENT] == {DocumentVisibility.CLIENT_PRIVATE}


# ---------------------------------------------------------------------------
# validate_document_visibility
# ---------------------------------------------------------------------------


class TestValidateVisibility:
    def test_internal_user_can_use_internal_private(self):
        ctx = _internal_user()
        validate_document_visibility(DocumentVisibility.INTERNAL_PRIVATE, None, ctx)

    def test_internal_user_can_use_internal_group(self):
        ctx = _internal_user(group_ids=[10])
        validate_document_visibility(DocumentVisibility.INTERNAL_GROUP, 10, ctx)

    def test_internal_user_cannot_use_client_private(self):
        ctx = _internal_user()
        with pytest.raises(Exception):  # noqa: B017
            validate_document_visibility(DocumentVisibility.CLIENT_PRIVATE, None, ctx)

    def test_client_user_can_use_client_private(self):
        ctx = _client_user()
        validate_document_visibility(DocumentVisibility.CLIENT_PRIVATE, None, ctx)

    def test_client_user_cannot_use_internal_public(self):
        ctx = _client_user()
        with pytest.raises(Exception):  # noqa: B017
            validate_document_visibility(DocumentVisibility.INTERNAL_PUBLIC, None, ctx)

    def test_non_admin_cannot_publish_internal_public(self):
        ctx = _internal_user(role=UserRole.USER)
        with pytest.raises(Exception):  # noqa: B017
            validate_document_visibility(DocumentVisibility.INTERNAL_PUBLIC, None, ctx)

    def test_admin_can_publish_internal_public(self):
        ctx = _internal_user(role=UserRole.ADMIN)
        validate_document_visibility(DocumentVisibility.INTERNAL_PUBLIC, None, ctx)

    def test_internal_group_requires_group_id(self):
        ctx = _internal_user()
        with pytest.raises(Exception):  # noqa: B017
            validate_document_visibility(DocumentVisibility.INTERNAL_GROUP, None, ctx)

    def test_internal_group_rejects_non_member(self):
        ctx = _internal_user(group_ids=[1, 2])
        with pytest.raises(Exception):  # noqa: B017
            validate_document_visibility(DocumentVisibility.INTERNAL_GROUP, 99, ctx)

    def test_internal_group_admin_bypasses_membership(self):
        ctx = _internal_user(role=UserRole.ADMIN, group_ids=[1, 2])
        validate_document_visibility(DocumentVisibility.INTERNAL_GROUP, 99, ctx)


# ---------------------------------------------------------------------------
# compute_owner_and_group
# ---------------------------------------------------------------------------


class TestOwnerAndGroup:
    def test_internal_public_returns_none_none(self):
        owner, group = compute_owner_and_group(DocumentVisibility.INTERNAL_PUBLIC, None, 1)
        assert owner is None
        assert group is None

    def test_internal_group_returns_none_group_id(self):
        owner, group = compute_owner_and_group(DocumentVisibility.INTERNAL_GROUP, 42, 1)
        assert owner is None
        assert group == 42

    def test_internal_private_returns_owner(self):
        owner, group = compute_owner_and_group(DocumentVisibility.INTERNAL_PRIVATE, None, 7)
        assert owner == 7
        assert group is None

    def test_client_private_returns_owner(self):
        owner, group = compute_owner_and_group(DocumentVisibility.CLIENT_PRIVATE, None, 99)
        assert owner == 99
        assert group is None


# ---------------------------------------------------------------------------
# can_view_document
# ---------------------------------------------------------------------------


class TestCanViewDocument:
    def test_internal_user_views_internal_public(self):
        doc = SimpleNamespace(visibility="internal_public", owner_id=1, group_id=None)
        assert can_view_document(doc, _internal_user()) is True

    def test_client_user_rejected_from_internal_public(self):
        doc = SimpleNamespace(visibility="internal_public", owner_id=1, group_id=None)
        assert can_view_document(doc, _client_user()) is False

    def test_internal_user_views_group_doc_if_member(self):
        doc = SimpleNamespace(visibility="internal_group", owner_id=None, group_id=10)
        assert can_view_document(doc, _internal_user(group_ids=[10])) is True

    def test_internal_user_rejected_from_group_doc_if_not_member(self):
        doc = SimpleNamespace(visibility="internal_group", owner_id=None, group_id=10)
        assert can_view_document(doc, _internal_user(group_ids=[1, 2])) is False

    def test_client_rejected_from_internal_group(self):
        doc = SimpleNamespace(visibility="internal_group", owner_id=None, group_id=1)
        assert can_view_document(doc, _client_user()) is False

    def test_internal_owner_views_private_doc(self):
        doc = SimpleNamespace(visibility="internal_private", owner_id=5, group_id=None)
        assert can_view_document(doc, _internal_user(user_id=5)) is True

    def test_internal_non_owner_rejected_from_private_doc(self):
        doc = SimpleNamespace(visibility="internal_private", owner_id=5, group_id=None)
        assert can_view_document(doc, _internal_user(user_id=9)) is False

    def test_client_views_own_private_doc(self):
        doc = SimpleNamespace(visibility="client_private", owner_id=50, group_id=None)
        assert can_view_document(doc, _client_user(user_id=50)) is True

    def test_client_rejected_from_other_client_doc(self):
        doc = SimpleNamespace(visibility="client_private", owner_id=51, group_id=None)
        assert can_view_document(doc, _client_user(user_id=50)) is False

    def test_admin_views_client_doc(self):
        doc = SimpleNamespace(visibility="client_private", owner_id=50, group_id=None)
        assert can_view_document(doc, _internal_user(role=UserRole.ADMIN)) is True

    def test_non_admin_rejected_from_client_doc(self):
        doc = SimpleNamespace(visibility="client_private", owner_id=50, group_id=None)
        assert can_view_document(doc, _internal_user(user_id=1, group_ids=[50, 51])) is False

    def test_unknown_visibility_raises(self):
        doc = SimpleNamespace(visibility="nonexistent", owner_id=1, group_id=None)
        with pytest.raises(ValidationError):
            can_view_document(doc, _internal_user())


# ---------------------------------------------------------------------------
# build_qdrant_filter (infrastructure layer)
# ---------------------------------------------------------------------------


class TestBuildQdrantFilter:
    def test_client_gets_owner_filter(self):
        user = {"id": 42, "kind": "client"}
        f = build_qdrant_filter(user, [])
        # Single condition: must=[visibility, owner_id] wrapped in should
        assert f.should is not None
        assert len(f.should) == 1
        inner = f.should[0]
        assert inner.must is not None
        assert len(inner.must) == 2

    def test_client_filter_has_correct_visibility(self):
        f = build_qdrant_filter({"id": 1, "kind": "client"}, [])
        inner = f.should[0]
        vis_match = inner.must[0]
        assert vis_match.match.value == "client_private"

    def test_internal_user_base_filter_has_public_and_private(self):
        f = build_qdrant_filter({"id": 1, "kind": "internal"}, [])
        assert f.should is not None
        assert len(f.should) == 2  # public + private

    def test_internal_with_groups_adds_group_filter(self):
        f = build_qdrant_filter({"id": 1, "kind": "internal"}, [10, 20])
        assert len(f.should) == 3  # public + private + group

    def test_internal_with_clients_adds_client_filter(self):
        f = build_qdrant_filter({"id": 1, "kind": "internal"}, [])
        # for_list=False: client_private NOT included in RAG search filters
        assert len(f.should) == 2  # public + private

    def test_internal_with_both_groups_and_clients(self):
        f = build_qdrant_filter({"id": 1, "kind": "internal"}, [10])
        # for_list=False: client_private NOT included, but group IS included
        assert len(f.should) == 3  # public + private + group


# ---------------------------------------------------------------------------
# A5: Invariant — is_in_search_scope == build_qdrant_filter conditions
# ---------------------------------------------------------------------------


class TestACLInvariant:
    """Property-based invariant: is_in_search_scope(doc, ctx) ⇔ build_qdrant_filter.

    For every combination of (role × visibility × owner × group), the domain
    rule `is_in_search_scope` must agree with the Qdrant filter structure
    produced by `build_qdrant_filter`.

    This invariant is the single most important correctness property of the
    ACL system. If it ever breaks, the RAG pipeline will either leak data
    (filter too permissive) or lose recall (filter too restrictive).
    """

    def _filter_allows(self, filter_obj, doc):
        """Check if a Qdrant Filter would match a document.

        Translates Qdrant Filter conditions into a simple match check.
        This is a simplified evaluator — enough for characterization tests.
        """
        if not filter_obj or not filter_obj.should:
            return False
        for condition in filter_obj.should:
            if self._condition_matches(condition, doc):
                return True
        return False

    def _condition_matches(self, condition, doc):
        """Check if a single Qdrant Filter condition matches a document."""
        # If condition is a Filter (nested), check its must conditions
        if hasattr(condition, "must") and condition.must and not hasattr(condition, "match"):
            return all(self._condition_matches(m, doc) for m in condition.must)
        # FieldCondition with match
        if hasattr(condition, "match"):
            key = condition.key
            match = condition.match
            # MatchValue
            if hasattr(match, "value"):
                if key == "metadata.visibility":
                    return doc.visibility == match.value
                if key == "metadata.owner_id":
                    return doc.owner_id == match.value
                if key == "metadata.group_id":
                    return doc.group_id == match.value
            # MatchAny
            if hasattr(match, "any"):
                if key == "metadata.owner_id":
                    return doc.owner_id in match.any
                if key == "metadata.group_id":
                    return doc.group_id in match.any
        return False

    def test_user_invariant(self):
        """USER role: is_in_search_scope matches filter conditions."""
        ctx = UserContext(user_id=1, user_kind="internal", user_role="user")
        user = {"id": 1, "kind": "internal", "role": "user"}
        f = build_qdrant_filter(user, [])

        docs = [
            SimpleNamespace(visibility="internal_public", owner_id=None, group_id=None),
            SimpleNamespace(visibility="internal_private", owner_id=1, group_id=None),
            SimpleNamespace(visibility="internal_private", owner_id=99, group_id=None),
            SimpleNamespace(visibility="internal_group", owner_id=None, group_id=5),
            SimpleNamespace(visibility="client_private", owner_id=1, group_id=None),
        ]
        for doc in docs:
            scope_result = is_in_search_scope(doc, ctx)
            filter_result = self._filter_allows(f, doc)
            assert scope_result == filter_result, (
                f"Mismatch for {doc.visibility} owner={doc.owner_id}: "
                f"is_in_search_scope={scope_result}, filter={filter_result}"
            )

    def test_admin_invariant(self):
        """ADMIN role: is_in_search_scope matches filter (no CLIENT_PRIVATE bonus)."""
        ctx = UserContext(user_id=1, user_kind="internal", user_role="admin")
        user = {"id": 1, "kind": "internal", "role": "admin"}
        f = build_qdrant_filter(user, [])

        docs = [
            SimpleNamespace(visibility="internal_public", owner_id=None, group_id=None),
            SimpleNamespace(visibility="internal_private", owner_id=1, group_id=None),
            SimpleNamespace(visibility="internal_private", owner_id=99, group_id=None),
            SimpleNamespace(visibility="client_private", owner_id=100, group_id=None),
        ]
        for doc in docs:
            scope_result = is_in_search_scope(doc, ctx)
            filter_result = self._filter_allows(f, doc)
            assert scope_result == filter_result, (
                f"Mismatch for {doc.visibility} owner={doc.owner_id}: "
                f"is_in_search_scope={scope_result}, filter={filter_result}"
            )

    def test_curator_no_managed_invariant(self):
        """CURATOR without managed_ids: same as USER."""
        ctx = UserContext(user_id=1, user_kind="internal", user_role="curator")
        user = {"id": 1, "kind": "internal", "role": "curator"}
        f = build_qdrant_filter(user, [])

        docs = [
            SimpleNamespace(visibility="internal_public", owner_id=None, group_id=None),
            SimpleNamespace(visibility="internal_private", owner_id=1, group_id=None),
            SimpleNamespace(visibility="internal_private", owner_id=99, group_id=None),
            SimpleNamespace(visibility="client_private", owner_id=100, group_id=None),
        ]
        for doc in docs:
            scope_result = is_in_search_scope(doc, ctx)
            filter_result = self._filter_allows(f, doc)
            assert scope_result == filter_result, (
                f"Mismatch for {doc.visibility} owner={doc.owner_id}: "
                f"is_in_search_scope={scope_result}, filter={filter_result}"
            )

    def test_curator_with_managed_invariant(self):
        """CURATOR with managed_ids: is_in_search_scope matches filter WITH managed-ids.

        NOTE: Today build_qdrant_filter is called WITHOUT managed-ids in the
        RAG pipeline. This test uses the CORRECT call (with managed-ids) to
        verify the invariant holds when Phase B fixes the gap.
        """
        ctx = UserContext(
            user_id=1,
            user_kind="internal",
            user_role="curator",
            managed_client_ids=[100],
            managed_internal_ids=[200],
        )
        user = {"id": 1, "kind": "internal", "role": "curator"}
        f = build_qdrant_filter(
            user,
            [],
            managed_client_ids=[100],
            managed_internal_ids=[200],
        )

        docs = [
            SimpleNamespace(visibility="internal_public", owner_id=None, group_id=None),
            SimpleNamespace(visibility="internal_private", owner_id=1, group_id=None),
            SimpleNamespace(visibility="internal_private", owner_id=200, group_id=None),
            SimpleNamespace(visibility="internal_private", owner_id=99, group_id=None),
            SimpleNamespace(visibility="client_private", owner_id=100, group_id=None),
            SimpleNamespace(visibility="client_private", owner_id=300, group_id=None),
        ]
        for doc in docs:
            scope_result = is_in_search_scope(doc, ctx)
            filter_result = self._filter_allows(f, doc)
            assert scope_result == filter_result, (
                f"Mismatch for {doc.visibility} owner={doc.owner_id}: "
                f"is_in_search_scope={scope_result}, filter={filter_result}"
            )

    def test_client_invariant(self):
        """CLIENT role: only own client_private."""
        ctx = UserContext(user_id=100, user_kind="client", user_role="user")
        user = {"id": 100, "kind": "client", "role": "user"}
        f = build_qdrant_filter(user, [])

        docs = [
            SimpleNamespace(visibility="client_private", owner_id=100, group_id=None),
            SimpleNamespace(visibility="client_private", owner_id=200, group_id=None),
            SimpleNamespace(visibility="internal_public", owner_id=None, group_id=None),
        ]
        for doc in docs:
            scope_result = is_in_search_scope(doc, ctx)
            filter_result = self._filter_allows(f, doc)
            assert scope_result == filter_result, (
                f"Mismatch for {doc.visibility} owner={doc.owner_id}: "
                f"is_in_search_scope={scope_result}, filter={filter_result}"
            )

    def test_curator_managed_client_invariant(self):
        """CURATOR with managed_client_ids: sees assigned CLIENT_PRIVATE."""
        ctx = UserContext(
            user_id=1,
            user_kind="internal",
            user_role="curator",
            managed_client_ids=[100],
        )
        user = {"id": 1, "kind": "internal", "role": "curator"}
        f = build_qdrant_filter(user, [], managed_client_ids=[100])

        docs = [
            SimpleNamespace(visibility="client_private", owner_id=100, group_id=None),
            SimpleNamespace(visibility="client_private", owner_id=200, group_id=None),
            SimpleNamespace(visibility="internal_public", owner_id=None, group_id=None),
            SimpleNamespace(visibility="internal_private", owner_id=1, group_id=None),
        ]
        for doc in docs:
            scope_result = is_in_search_scope(doc, ctx)
            filter_result = self._filter_allows(f, doc)
            assert scope_result == filter_result, (
                f"Mismatch for {doc.visibility} owner={doc.owner_id}: "
                f"is_in_search_scope={scope_result}, filter={filter_result}"
            )

    def test_curator_managed_internal_invariant(self):
        """CURATOR with managed_internal_ids: sees assigned INTERNAL_PRIVATE."""
        ctx = UserContext(
            user_id=1,
            user_kind="internal",
            user_role="curator",
            managed_internal_ids=[20],
        )
        user = {"id": 1, "kind": "internal", "role": "curator"}
        f = build_qdrant_filter(user, [], managed_internal_ids=[20])

        docs = [
            SimpleNamespace(visibility="internal_private", owner_id=20, group_id=None),
            SimpleNamespace(visibility="internal_private", owner_id=30, group_id=None),
            SimpleNamespace(visibility="internal_private", owner_id=1, group_id=None),
            SimpleNamespace(visibility="internal_public", owner_id=None, group_id=None),
        ]
        for doc in docs:
            scope_result = is_in_search_scope(doc, ctx)
            filter_result = self._filter_allows(f, doc)
            assert scope_result == filter_result, (
                f"Mismatch for {doc.visibility} owner={doc.owner_id}: "
                f"is_in_search_scope={scope_result}, filter={filter_result}"
            )

    def test_curator_managed_group_invariant(self):
        """CURATOR with managed_group_ids: sees assigned group docs."""
        ctx = UserContext(
            user_id=1,
            user_kind="internal",
            user_role="curator",
            managed_group_ids=[5],
        )
        user = {"id": 1, "kind": "internal", "role": "curator"}
        f = build_qdrant_filter(user, [], managed_group_ids=[5])

        docs = [
            SimpleNamespace(visibility="internal_group", owner_id=None, group_id=5),
            SimpleNamespace(visibility="internal_group", owner_id=None, group_id=99),
            SimpleNamespace(visibility="internal_public", owner_id=None, group_id=None),
        ]
        for doc in docs:
            scope_result = is_in_search_scope(doc, ctx)
            filter_result = self._filter_allows(f, doc)
            assert scope_result == filter_result, (
                f"Mismatch for {doc.visibility} owner={doc.owner_id} group={doc.group_id}: "
                f"is_in_search_scope={scope_result}, filter={filter_result}"
            )

    def test_curator_all_managed_invariant(self):
        """CURATOR with all managed IDs: full scope."""
        ctx = UserContext(
            user_id=1,
            user_kind="internal",
            user_role="curator",
            group_ids=[5],
            managed_client_ids=[100],
            managed_internal_ids=[20],
            managed_group_ids=[15],
        )
        user = {"id": 1, "kind": "internal", "role": "curator"}
        f = build_qdrant_filter(
            user,
            [5],
            managed_client_ids=[100],
            managed_internal_ids=[20],
            managed_group_ids=[15],
        )

        docs = [
            SimpleNamespace(visibility="internal_public", owner_id=None, group_id=None),
            SimpleNamespace(visibility="internal_private", owner_id=1, group_id=None),
            SimpleNamespace(visibility="internal_private", owner_id=20, group_id=None),
            SimpleNamespace(visibility="internal_private", owner_id=99, group_id=None),
            SimpleNamespace(visibility="client_private", owner_id=100, group_id=None),
            SimpleNamespace(visibility="client_private", owner_id=999, group_id=None),
            SimpleNamespace(visibility="internal_group", owner_id=None, group_id=5),
            SimpleNamespace(visibility="internal_group", owner_id=None, group_id=15),
            SimpleNamespace(visibility="internal_group", owner_id=None, group_id=99),
        ]
        for doc in docs:
            scope_result = is_in_search_scope(doc, ctx)
            filter_result = self._filter_allows(f, doc)
            assert scope_result == filter_result, (
                f"Mismatch for {doc.visibility} owner={doc.owner_id} group={doc.group_id}: "
                f"is_in_search_scope={scope_result}, filter={filter_result}"
            )

    def test_admin_with_groups_invariant(self):
        """ADMIN with groups: bypasses group membership, no CLIENT_PRIVATE in search."""
        ctx = UserContext(
            user_id=1,
            user_kind="internal",
            user_role="admin",
            group_ids=[5],
        )
        user = {"id": 1, "kind": "internal", "role": "admin"}
        f = build_qdrant_filter(user, [5])

        docs = [
            SimpleNamespace(visibility="internal_public", owner_id=None, group_id=None),
            SimpleNamespace(visibility="internal_private", owner_id=1, group_id=None),
            SimpleNamespace(visibility="internal_private", owner_id=99, group_id=None),
            SimpleNamespace(visibility="internal_group", owner_id=None, group_id=5),
            SimpleNamespace(visibility="internal_group", owner_id=None, group_id=99),
            SimpleNamespace(visibility="client_private", owner_id=100, group_id=None),
        ]
        for doc in docs:
            scope_result = is_in_search_scope(doc, ctx)
            filter_result = self._filter_allows(f, doc)
            assert scope_result == filter_result, (
                f"Mismatch for {doc.visibility} owner={doc.owner_id} group={doc.group_id}: "
                f"is_in_search_scope={scope_result}, filter={filter_result}"
            )

    def test_conditions_count_matches_role(self):
        """Number of filter conditions matches expected visibility conditions.

        Base conditions (INTERNAL): PUBLIC + PRIVATE(self) = 2
        Role adds:
          USER: +0
          ADMIN (search): +GROUP(bypass) + PRIVATE(bypass) = +2
          CURATOR no managed: +0
          CURATOR with managed: +PRIVATE(assigned) + CLIENT_PRIVATE(assigned) = +2
        """
        # USER: public + private(self) = 2
        f_user = build_qdrant_filter({"id": 1, "kind": "internal", "role": "user"}, [])
        assert len(f_user.should) == 2

        # ADMIN (search): public + private(self) + group(bypass) + private(bypass) = 4
        f_admin = build_qdrant_filter({"id": 1, "kind": "internal", "role": "admin"}, [])
        assert len(f_admin.should) == 4

        # CURATOR no managed: public + private(self) = 2
        f_curator = build_qdrant_filter({"id": 1, "kind": "internal", "role": "curator"}, [])
        assert len(f_curator.should) == 2

        # CURATOR with managed: public + private(self) + private(assigned) + client(assigned) = 4
        f_curator_managed = build_qdrant_filter(
            {"id": 1, "kind": "internal", "role": "curator"},
            [],
            managed_client_ids=[100],
            managed_internal_ids=[200],
        )
        assert len(f_curator_managed.should) == 4

    def test_visibility_conditions_vs_filter_always_consistent(self):
        """Meta-test: get_visibility_conditions(for_list=False) count == filter conditions count.

        Both use the same canonical IR (VisibilityCondition). The filter
        translates each condition into a Qdrant Filter clause. Count must match.
        """
        test_cases = [
            ("internal", 1, "user", [], {}),
            ("internal", 1, "admin", [], {}),
            ("internal", 1, "curator", [], {}),
            ("internal", 1, "curator", [], {"managed_client_ids": [100]}),
            ("internal", 1, "curator", [], {"managed_client_ids": [100], "managed_internal_ids": [200]}),
            ("internal", 1, "curator", [5, 10], {"managed_group_ids": [15]}),
            ("client", 100, "user", [], {}),
        ]
        for kind, uid, role, groups, managed in test_cases:
            conds = get_visibility_conditions(
                UserKind(kind),
                uid,
                groups,
                for_list=False,
                user_role=UserRole(role),
                **managed,
            )
            user_dict = {"id": uid, "kind": kind, "role": role}
            f = build_qdrant_filter(
                user_dict,
                groups,
                managed_client_ids=managed.get("managed_client_ids"),
                managed_internal_ids=managed.get("managed_internal_ids"),
                managed_group_ids=managed.get("managed_group_ids"),
            )
            assert len(conds) == len(f.should), (
                f"Condition count mismatch for {kind}/{role}: "
                f"visibility_conditions={len(conds)}, filter={len(f.should)}"
            )
