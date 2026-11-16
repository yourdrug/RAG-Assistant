"""Tests for access control logic.

Domain rules: domain/knowledge/access_control.py
Qdrant filter: infrastructure/acl.py
"""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

from qdrant_client import models
from qdrant_client.local.payload_filters import check_filter
from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine, or_, select

from domain.identity.roles import UserKind, UserRole
from domain.knowledge.document import Document
from domain.knowledge.enums import DocumentVisibility
from domain.shared.exceptions import BusinessRuleViolationError
from infrastructure.repositories.sql_acl import render_visibility_conditions


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import pytest

from domain.identity.user_context import UserContext
from domain.knowledge.access_control import (
    ALLOWED_VISIBILITY_FOR_KIND,
    can_view_document,
    compute_owner_and_group,
    get_visibility_conditions,
    is_in_search_scope,
    validate_document_visibility,
)
from domain.shared.exceptions import ValidationError
from infrastructure.repositories.vector.acl import build_qdrant_filter


if TYPE_CHECKING:
    from sqlalchemy.orm import InstrumentedAttribute


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def internal_user(
    user_id: int = 1, role: UserRole = UserRole.USER, group_ids: list[int] | None = None
) -> UserContext:
    return UserContext(
        user_id=user_id,
        user_kind=UserKind.INTERNAL,
        user_role=role,
        group_ids=tuple(group_ids or []),
    )


def client_user(user_id: int = 100, group_ids: list[int] | None = None) -> UserContext:
    return UserContext(
        user_id=user_id,
        user_kind=UserKind.CLIENT,
        user_role=UserRole.USER,
        group_ids=tuple(group_ids or []),
    )


# ---------------------------------------------------------------------------
# ALLOWED_VISIBILITY_FOR_KIND
# ---------------------------------------------------------------------------


class TestAllowedVisibility:
    def test_internal_allowed_visibilities(self) -> None:
        assert ALLOWED_VISIBILITY_FOR_KIND[UserKind.INTERNAL] == {
            DocumentVisibility.INTERNAL_PUBLIC,
            DocumentVisibility.INTERNAL_GROUP,
            DocumentVisibility.INTERNAL_PRIVATE,
            DocumentVisibility.CLIENT_PRIVATE,
            DocumentVisibility.CLIENT_PUBLIC,
        }

    def test_client_allowed_visibilities(self) -> None:
        assert ALLOWED_VISIBILITY_FOR_KIND[UserKind.CLIENT] == {DocumentVisibility.CLIENT_PRIVATE}


# ---------------------------------------------------------------------------
# validate_document_visibility
# ---------------------------------------------------------------------------


class TestValidateVisibility:
    def test_internal_user_can_use_internal_private(self) -> None:
        ctx = internal_user()
        validate_document_visibility(DocumentVisibility.INTERNAL_PRIVATE, None, ctx)

    def test_internal_user_can_use_internal_group(self) -> None:
        ctx = internal_user(group_ids=[10])
        validate_document_visibility(DocumentVisibility.INTERNAL_GROUP, 10, ctx)

    def test_internal_user_cannot_use_client_private(self) -> None:
        ctx = internal_user()
        with pytest.raises(BusinessRuleViolationError, match="Only admin can upload"):
            validate_document_visibility(DocumentVisibility.CLIENT_PRIVATE, None, ctx)

    def test_client_user_can_use_client_private(self) -> None:
        ctx = client_user()
        validate_document_visibility(DocumentVisibility.CLIENT_PRIVATE, None, ctx)

    def test_client_user_cannot_use_internal_public(self) -> None:
        ctx = client_user()
        with pytest.raises(ValidationError, match="not available for kind"):
            validate_document_visibility(DocumentVisibility.INTERNAL_PUBLIC, None, ctx)

    def test_non_admin_cannot_publish_internal_public(self) -> None:
        ctx = internal_user(role=UserRole.USER)
        with pytest.raises(BusinessRuleViolationError, match="Only admin can publish"):
            validate_document_visibility(DocumentVisibility.INTERNAL_PUBLIC, None, ctx)

    def test_admin_can_publish_internal_public(self) -> None:
        ctx = internal_user(role=UserRole.ADMIN)
        validate_document_visibility(DocumentVisibility.INTERNAL_PUBLIC, None, ctx)

    def test_internal_group_requires_group_id(self) -> None:
        ctx = internal_user()
        with pytest.raises(ValidationError, match="group_id required"):
            validate_document_visibility(DocumentVisibility.INTERNAL_GROUP, None, ctx)

    def test_internal_group_rejects_non_member(self) -> None:
        ctx = internal_user(group_ids=[1, 2])
        with pytest.raises(BusinessRuleViolationError, match="not a member"):
            validate_document_visibility(DocumentVisibility.INTERNAL_GROUP, 99, ctx)

    def test_internal_group_admin_bypasses_membership(self) -> None:
        ctx = internal_user(role=UserRole.ADMIN, group_ids=[1, 2])
        validate_document_visibility(DocumentVisibility.INTERNAL_GROUP, 99, ctx)


# ---------------------------------------------------------------------------
# compute_owner_and_group
# ---------------------------------------------------------------------------


class TestOwnerAndGroup:
    def test_internal_public_returns_none_none(self) -> None:
        owner, group = compute_owner_and_group(DocumentVisibility.INTERNAL_PUBLIC, None, 1)
        assert owner is None
        assert group is None

    def test_internal_group_returns_none_group_id(self) -> None:
        owner, group = compute_owner_and_group(DocumentVisibility.INTERNAL_GROUP, 42, 1)
        assert owner is None
        assert group == 42

    def test_internal_private_returns_owner(self) -> None:
        owner, group = compute_owner_and_group(DocumentVisibility.INTERNAL_PRIVATE, None, 7)
        assert owner == 7
        assert group is None

    def test_client_private_returns_owner(self) -> None:
        owner, group = compute_owner_and_group(DocumentVisibility.CLIENT_PRIVATE, None, 99)
        assert owner == 99
        assert group is None


# ---------------------------------------------------------------------------
# can_view_document
# ---------------------------------------------------------------------------


class TestCanViewDocument:
    def test_internal_user_views_internal_public(self) -> None:
        doc = Document(visibility=DocumentVisibility.INTERNAL_PUBLIC, owner_id=1, group_id=None)
        assert can_view_document(doc, internal_user()) is True

    def test_client_user_rejected_from_internal_public(self) -> None:
        doc = Document(visibility=DocumentVisibility.INTERNAL_PUBLIC, owner_id=1, group_id=None)
        assert can_view_document(doc, client_user()) is False

    def test_internal_user_views_group_doc_if_member(self) -> None:
        doc = Document(visibility=DocumentVisibility.INTERNAL_GROUP, owner_id=None, group_id=10)
        assert can_view_document(doc, internal_user(group_ids=[10])) is True

    def test_internal_user_rejected_from_group_doc_if_not_member(self) -> None:
        doc = Document(visibility=DocumentVisibility.INTERNAL_GROUP, owner_id=None, group_id=10)
        assert can_view_document(doc, internal_user(group_ids=[1, 2])) is False

    def test_client_rejected_from_internal_group(self) -> None:
        doc = Document(visibility=DocumentVisibility.INTERNAL_GROUP, owner_id=None, group_id=1)
        assert can_view_document(doc, client_user()) is False

    def test_internal_owner_views_private_doc(self) -> None:
        doc = Document(visibility=DocumentVisibility.INTERNAL_PRIVATE, owner_id=5, group_id=None)
        assert can_view_document(doc, internal_user(user_id=5)) is True

    def test_internal_non_owner_rejected_from_private_doc(self) -> None:
        doc = Document(visibility=DocumentVisibility.INTERNAL_PRIVATE, owner_id=5, group_id=None)
        assert can_view_document(doc, internal_user(user_id=9)) is False

    def test_client_views_own_private_doc(self) -> None:
        doc = Document(visibility=DocumentVisibility.CLIENT_PRIVATE, owner_id=50, group_id=None)
        assert can_view_document(doc, client_user(user_id=50)) is True

    def test_client_rejected_from_other_client_doc(self) -> None:
        doc = Document(visibility=DocumentVisibility.CLIENT_PRIVATE, owner_id=51, group_id=None)
        assert can_view_document(doc, client_user(user_id=50)) is False

    def test_admin_views_client_doc(self) -> None:
        doc = Document(visibility=DocumentVisibility.CLIENT_PRIVATE, owner_id=50, group_id=None)
        assert can_view_document(doc, internal_user(role=UserRole.ADMIN)) is True

    def test_non_admin_rejected_from_client_doc(self) -> None:
        doc = Document(visibility=DocumentVisibility.CLIENT_PRIVATE, owner_id=50, group_id=None)
        assert can_view_document(doc, internal_user(user_id=1, group_ids=[50, 51])) is False

    def test_unknown_visibility_raises(self) -> None:
        doc = SimpleNamespace(visibility="nonexistent", owner_id=1, group_id=None)
        with pytest.raises(ValidationError):
            can_view_document(cast("Document", doc), internal_user())


# ---------------------------------------------------------------------------
# build_qdrant_filter (infrastructure layer)
# ---------------------------------------------------------------------------


class TestBuildQdrantFilter:
    def test_client_gets_owner_filter(self) -> None:
        user = {"id": 42, "kind": "client"}
        f = build_qdrant_filter(user, [])
        assert isinstance(f.should, list)
        # Private-owner clause plus common client articles.
        assert f.should is not None
        assert len(f.should) == 2
        inner = f.should[0]
        assert isinstance(inner, models.Filter)
        assert isinstance(inner.must, list)
        assert inner.must is not None
        assert len(inner.must) == 2

    def test_client_filter_has_correct_visibility(self) -> None:
        f = build_qdrant_filter({"id": 1, "kind": "client"}, [])
        assert isinstance(f.should, list)
        inner = f.should[0]
        assert isinstance(inner, models.Filter)
        assert isinstance(inner.must, list)
        vis_match = inner.must[0]
        assert isinstance(vis_match, models.FieldCondition)
        assert isinstance(vis_match.match, models.MatchValue)
        assert vis_match.match.value == "client_private"

    def test_internal_user_base_filter_has_public_and_private(self) -> None:
        f = build_qdrant_filter({"id": 1, "kind": "internal"}, [])
        assert isinstance(f.should, list)
        assert f.should is not None
        assert len(f.should) == 3  # internal public + private + client public

    def test_internal_with_groups_adds_group_filter(self) -> None:
        f = build_qdrant_filter({"id": 1, "kind": "internal"}, [10, 20])
        assert isinstance(f.should, list)
        assert len(f.should) == 4  # internal public + private + group + client public

    def test_internal_with_clients_adds_client_filter(self) -> None:
        f = build_qdrant_filter({"id": 1, "kind": "internal"}, [])
        assert isinstance(f.should, list)
        # for_list=False: client_private NOT included in RAG search filters
        assert len(f.should) == 3  # internal public + private + client public

    def test_internal_with_both_groups_and_clients(self) -> None:
        f = build_qdrant_filter({"id": 1, "kind": "internal"}, [10])
        assert isinstance(f.should, list)
        # for_list=False: client_private NOT included, but group IS included
        assert len(f.should) == 4  # internal public + private + group + client public

    def test_user_context_client_gets_owner_filter(self) -> None:
        ctx = UserContext(user_id=42, user_kind=UserKind.CLIENT, user_role=UserRole.USER)
        f = build_qdrant_filter(ctx, [])
        assert isinstance(f.should, list)
        assert f.should is not None
        assert len(f.should) == 2
        inner = f.should[0]
        assert isinstance(inner, models.Filter)
        assert isinstance(inner.must, list)
        assert inner.must is not None
        assert len(inner.must) == 2
        owner = inner.must[1]
        assert isinstance(owner, models.FieldCondition)
        assert isinstance(owner.match, models.MatchValue)
        assert owner.match.value == 42

    def test_user_context_and_dict_produce_identical_filter(self) -> None:
        ctx = UserContext(
            user_id=7,
            user_kind=UserKind.INTERNAL,
            user_role=UserRole.USER,
            group_ids=(10, 20),
        )
        f_obj = build_qdrant_filter(ctx, [10, 20])
        assert isinstance(f_obj.should, list)
        f_dict = build_qdrant_filter({"id": 7, "kind": "internal", "role": "user"}, [10, 20])
        assert isinstance(f_dict.should, list)
        assert f_obj.model_dump() == f_dict.model_dump()

    def test_user_context_with_role_builds_same_conditions(self) -> None:
        ctx = UserContext(user_id=1, user_kind=UserKind.INTERNAL, user_role=UserRole.CURATOR)
        f = build_qdrant_filter(
            ctx,
            [],
            managed_client_ids=[100],
            managed_internal_ids=[200],
            managed_group_ids=[300],
        )
        assert isinstance(f.should, list)
        assert f.should is not None
        assert len(f.should) > 0


# ---------------------------------------------------------------------------
# A5: Invariant — is_in_search_scope == build_qdrant_filter conditions
# ---------------------------------------------------------------------------


class TestACLInvariant:
    """Property-based invariant: is_in_search_scope(doc, ctx) ⇔ build_qdrant_filter.

    For every combination of (role / visibility / owner / group), the domain
    rule `is_in_search_scope` must agree with the Qdrant filter structure
    produced by `build_qdrant_filter`.

    This invariant is the single most important correctness property of the
    ACL system. If it ever breaks, the RAG pipeline will either leak data
    (filter too permissive) or lose recall (filter too restrictive).
    """

    def filter_allows(self, filter_obj: models.Filter, doc: Document) -> bool:
        if not isinstance(filter_obj.should, list):
            return False

        return any(self.condition_matches(condition, doc) for condition in filter_obj.should)

    def condition_matches(self, condition: object, doc: Document) -> bool:
        if isinstance(condition, models.Filter) and isinstance(condition.must, list):
            return all(self.condition_matches(item, doc) for item in condition.must)

        if not isinstance(condition, models.FieldCondition):
            return False

        values = {
            "metadata.visibility": doc.visibility,
            "metadata.owner_id": doc.owner_id,
            "metadata.group_id": doc.group_id,
        }
        value = values.get(condition.key)

        if isinstance(condition.match, models.MatchValue):
            return bool(value == condition.match.value)

        return isinstance(condition.match, models.MatchAny) and value in condition.match.any

    def test_user_invariant(self) -> None:
        """USER role: is_in_search_scope matches filter conditions."""
        ctx = UserContext(user_id=1, user_kind=UserKind.INTERNAL, user_role=UserRole.USER)
        user = {"id": 1, "kind": "internal", "role": "user"}
        f = build_qdrant_filter(user, [])
        assert isinstance(f.should, list)

        docs = [
            Document(visibility=DocumentVisibility.INTERNAL_PUBLIC, owner_id=None, group_id=None),
            Document(visibility=DocumentVisibility.INTERNAL_PRIVATE, owner_id=1, group_id=None),
            Document(visibility=DocumentVisibility.INTERNAL_PRIVATE, owner_id=99, group_id=None),
            Document(visibility=DocumentVisibility.INTERNAL_GROUP, owner_id=None, group_id=5),
            Document(visibility=DocumentVisibility.CLIENT_PRIVATE, owner_id=1, group_id=None),
        ]

        for doc in docs:
            scope_result = is_in_search_scope(doc, ctx)
            filter_result = self.filter_allows(f, doc)
            assert scope_result == filter_result, (
                f"Mismatch for {doc.visibility} owner={doc.owner_id}: "
                f"is_in_search_scope={scope_result}, filter={filter_result}"
            )

    def test_admin_invariant(self) -> None:
        """ADMIN role: is_in_search_scope matches filter (no CLIENT_PRIVATE bonus)."""
        ctx = UserContext(user_id=1, user_kind=UserKind.INTERNAL, user_role=UserRole.ADMIN)
        user = {"id": 1, "kind": "internal", "role": "admin"}
        f = build_qdrant_filter(user, [])
        assert isinstance(f.should, list)

        docs = [
            Document(visibility=DocumentVisibility.INTERNAL_PUBLIC, owner_id=None, group_id=None),
            Document(visibility=DocumentVisibility.INTERNAL_PRIVATE, owner_id=1, group_id=None),
            Document(visibility=DocumentVisibility.INTERNAL_PRIVATE, owner_id=99, group_id=None),
            Document(visibility=DocumentVisibility.CLIENT_PRIVATE, owner_id=100, group_id=None),
        ]

        for doc in docs:
            scope_result = is_in_search_scope(doc, ctx)
            filter_result = self.filter_allows(f, doc)
            assert scope_result == filter_result, (
                f"Mismatch for {doc.visibility} owner={doc.owner_id}: "
                f"is_in_search_scope={scope_result}, filter={filter_result}"
            )

    def test_curator_no_managed_invariant(self) -> None:
        """CURATOR without managed_ids: same as USER."""
        ctx = UserContext(user_id=1, user_kind=UserKind.INTERNAL, user_role=UserRole.CURATOR)
        user = {"id": 1, "kind": "internal", "role": "curator"}
        f = build_qdrant_filter(user, [])
        assert isinstance(f.should, list)

        docs = [
            Document(visibility=DocumentVisibility.INTERNAL_PUBLIC, owner_id=None, group_id=None),
            Document(visibility=DocumentVisibility.INTERNAL_PRIVATE, owner_id=1, group_id=None),
            Document(visibility=DocumentVisibility.INTERNAL_PRIVATE, owner_id=99, group_id=None),
            Document(visibility=DocumentVisibility.CLIENT_PRIVATE, owner_id=100, group_id=None),
        ]

        for doc in docs:
            scope_result = is_in_search_scope(doc, ctx)
            filter_result = self.filter_allows(f, doc)
            assert scope_result == filter_result, (
                f"Mismatch for {doc.visibility} owner={doc.owner_id}: "
                f"is_in_search_scope={scope_result}, filter={filter_result}"
            )

    def test_curator_with_managed_invariant(self) -> None:
        """CURATOR with managed_ids: is_in_search_scope matches filter WITH managed-ids.

        NOTE: Today build_qdrant_filter is called WITHOUT managed-ids in the
        RAG pipeline. This test uses the CORRECT call (with managed-ids) to
        verify the invariant holds when Phase B fixes the gap.
        """
        ctx = UserContext(
            user_id=1,
            user_kind=UserKind.INTERNAL,
            user_role=UserRole.CURATOR,
            managed_client_ids=(100,),
            managed_internal_ids=(200,),
        )
        user = {"id": 1, "kind": "internal", "role": "curator"}
        f = build_qdrant_filter(
            user,
            [],
            managed_client_ids=[100],
            managed_internal_ids=[200],
        )
        assert isinstance(f.should, list)

        docs = [
            Document(visibility=DocumentVisibility.INTERNAL_PUBLIC, owner_id=None, group_id=None),
            Document(visibility=DocumentVisibility.INTERNAL_PRIVATE, owner_id=1, group_id=None),
            Document(visibility=DocumentVisibility.INTERNAL_PRIVATE, owner_id=200, group_id=None),
            Document(visibility=DocumentVisibility.INTERNAL_PRIVATE, owner_id=99, group_id=None),
            Document(visibility=DocumentVisibility.CLIENT_PRIVATE, owner_id=100, group_id=None),
            Document(visibility=DocumentVisibility.CLIENT_PRIVATE, owner_id=300, group_id=None),
        ]

        for doc in docs:
            scope_result = is_in_search_scope(doc, ctx)
            filter_result = self.filter_allows(f, doc)
            assert scope_result == filter_result, (
                f"Mismatch for {doc.visibility} owner={doc.owner_id}: "
                f"is_in_search_scope={scope_result}, filter={filter_result}"
            )

    def test_client_invariant(self) -> None:
        """CLIENT role: only own client_private."""
        ctx = UserContext(user_id=100, user_kind=UserKind.CLIENT, user_role=UserRole.USER)
        user = {"id": 100, "kind": "client", "role": "user"}
        f = build_qdrant_filter(user, [])
        assert isinstance(f.should, list)

        docs = [
            Document(visibility=DocumentVisibility.CLIENT_PRIVATE, owner_id=100, group_id=None),
            Document(visibility=DocumentVisibility.CLIENT_PRIVATE, owner_id=200, group_id=None),
            Document(visibility=DocumentVisibility.INTERNAL_PUBLIC, owner_id=None, group_id=None),
        ]

        for doc in docs:
            scope_result = is_in_search_scope(doc, ctx)
            filter_result = self.filter_allows(f, doc)
            assert scope_result == filter_result, (
                f"Mismatch for {doc.visibility} owner={doc.owner_id}: "
                f"is_in_search_scope={scope_result}, filter={filter_result}"
            )

    def test_curator_managed_client_invariant(self) -> None:
        """CURATOR with managed_client_ids: sees assigned CLIENT_PRIVATE."""
        ctx = UserContext(
            user_id=1,
            user_kind=UserKind.INTERNAL,
            user_role=UserRole.CURATOR,
            managed_client_ids=(100,),
        )
        user = {"id": 1, "kind": "internal", "role": "curator"}
        f = build_qdrant_filter(user, [], managed_client_ids=[100])
        assert isinstance(f.should, list)

        docs = [
            Document(visibility=DocumentVisibility.CLIENT_PRIVATE, owner_id=100, group_id=None),
            Document(visibility=DocumentVisibility.CLIENT_PRIVATE, owner_id=200, group_id=None),
            Document(visibility=DocumentVisibility.INTERNAL_PUBLIC, owner_id=None, group_id=None),
            Document(visibility=DocumentVisibility.INTERNAL_PRIVATE, owner_id=1, group_id=None),
        ]

        for doc in docs:
            scope_result = is_in_search_scope(doc, ctx)
            filter_result = self.filter_allows(f, doc)
            assert scope_result == filter_result, (
                f"Mismatch for {doc.visibility} owner={doc.owner_id}: "
                f"is_in_search_scope={scope_result}, filter={filter_result}"
            )

    def test_curator_managed_internal_invariant(self) -> None:
        """CURATOR with managed_internal_ids: sees assigned INTERNAL_PRIVATE."""
        ctx = UserContext(
            user_id=1,
            user_kind=UserKind.INTERNAL,
            user_role=UserRole.CURATOR,
            managed_internal_ids=(20,),
        )
        user = {"id": 1, "kind": "internal", "role": "curator"}
        f = build_qdrant_filter(user, [], managed_internal_ids=[20])
        assert isinstance(f.should, list)

        docs = [
            Document(visibility=DocumentVisibility.INTERNAL_PRIVATE, owner_id=20, group_id=None),
            Document(visibility=DocumentVisibility.INTERNAL_PRIVATE, owner_id=30, group_id=None),
            Document(visibility=DocumentVisibility.INTERNAL_PRIVATE, owner_id=1, group_id=None),
            Document(visibility=DocumentVisibility.INTERNAL_PUBLIC, owner_id=None, group_id=None),
        ]

        for doc in docs:
            scope_result = is_in_search_scope(doc, ctx)
            filter_result = self.filter_allows(f, doc)
            assert scope_result == filter_result, (
                f"Mismatch for {doc.visibility} owner={doc.owner_id}: "
                f"is_in_search_scope={scope_result}, filter={filter_result}"
            )

    def test_curator_managed_group_invariant(self) -> None:
        """CURATOR with managed_group_ids: sees assigned group docs."""
        ctx = UserContext(
            user_id=1,
            user_kind=UserKind.INTERNAL,
            user_role=UserRole.CURATOR,
            managed_group_ids=(5,),
        )
        user = {"id": 1, "kind": "internal", "role": "curator"}
        f = build_qdrant_filter(user, [], managed_group_ids=[5])
        assert isinstance(f.should, list)

        docs = [
            Document(visibility=DocumentVisibility.INTERNAL_GROUP, owner_id=None, group_id=5),
            Document(visibility=DocumentVisibility.INTERNAL_GROUP, owner_id=None, group_id=99),
            Document(visibility=DocumentVisibility.INTERNAL_PUBLIC, owner_id=None, group_id=None),
        ]

        for doc in docs:
            scope_result = is_in_search_scope(doc, ctx)
            filter_result = self.filter_allows(f, doc)
            assert scope_result == filter_result, (
                f"Mismatch for {doc.visibility} owner={doc.owner_id} group={doc.group_id}: "
                f"is_in_search_scope={scope_result}, filter={filter_result}"
            )

    def test_curator_all_managed_invariant(self) -> None:
        """CURATOR with all managed IDs: full scope."""
        ctx = UserContext(
            user_id=1,
            user_kind=UserKind.INTERNAL,
            user_role=UserRole.CURATOR,
            group_ids=(5,),
            managed_client_ids=(100,),
            managed_internal_ids=(20,),
            managed_group_ids=(15,),
        )
        user = {"id": 1, "kind": "internal", "role": "curator"}
        f = build_qdrant_filter(
            user,
            [5],
            managed_client_ids=[100],
            managed_internal_ids=[20],
            managed_group_ids=[15],
        )
        assert isinstance(f.should, list)

        docs = [
            Document(visibility=DocumentVisibility.INTERNAL_PUBLIC, owner_id=None, group_id=None),
            Document(visibility=DocumentVisibility.INTERNAL_PRIVATE, owner_id=1, group_id=None),
            Document(visibility=DocumentVisibility.INTERNAL_PRIVATE, owner_id=20, group_id=None),
            Document(visibility=DocumentVisibility.INTERNAL_PRIVATE, owner_id=99, group_id=None),
            Document(visibility=DocumentVisibility.CLIENT_PRIVATE, owner_id=100, group_id=None),
            Document(visibility=DocumentVisibility.CLIENT_PRIVATE, owner_id=999, group_id=None),
            Document(visibility=DocumentVisibility.INTERNAL_GROUP, owner_id=None, group_id=5),
            Document(visibility=DocumentVisibility.INTERNAL_GROUP, owner_id=None, group_id=15),
            Document(visibility=DocumentVisibility.INTERNAL_GROUP, owner_id=None, group_id=99),
        ]

        for doc in docs:
            scope_result = is_in_search_scope(doc, ctx)
            filter_result = self.filter_allows(f, doc)
            assert scope_result == filter_result, (
                f"Mismatch for {doc.visibility} owner={doc.owner_id} group={doc.group_id}: "
                f"is_in_search_scope={scope_result}, filter={filter_result}"
            )

    def test_admin_with_groups_invariant(self) -> None:
        """ADMIN with groups: bypasses group membership, no CLIENT_PRIVATE in search."""
        ctx = UserContext(
            user_id=1,
            user_kind=UserKind.INTERNAL,
            user_role=UserRole.ADMIN,
            group_ids=(5,),
        )
        user = {"id": 1, "kind": "internal", "role": "admin"}
        f = build_qdrant_filter(user, [5])
        assert isinstance(f.should, list)

        docs = [
            Document(visibility=DocumentVisibility.INTERNAL_PUBLIC, owner_id=None, group_id=None),
            Document(visibility=DocumentVisibility.INTERNAL_PRIVATE, owner_id=1, group_id=None),
            Document(visibility=DocumentVisibility.INTERNAL_PRIVATE, owner_id=99, group_id=None),
            Document(visibility=DocumentVisibility.INTERNAL_GROUP, owner_id=None, group_id=5),
            Document(visibility=DocumentVisibility.INTERNAL_GROUP, owner_id=None, group_id=99),
            Document(visibility=DocumentVisibility.CLIENT_PRIVATE, owner_id=100, group_id=None),
        ]

        for doc in docs:
            scope_result = is_in_search_scope(doc, ctx)
            filter_result = self.filter_allows(f, doc)
            assert scope_result == filter_result, (
                f"Mismatch for {doc.visibility} owner={doc.owner_id} group={doc.group_id}: "
                f"is_in_search_scope={scope_result}, filter={filter_result}"
            )

    def test_conditions_count_matches_role(self) -> None:
        """Number of filter conditions matches expected visibility conditions.

        Base conditions (INTERNAL): PUBLIC + PRIVATE(self) = 2
        Role adds:
          USER: +0
          ADMIN (search): +GROUP(bypass) + PRIVATE(bypass) = +2
          CURATOR no managed: +0
          CURATOR with managed: +PRIVATE(assigned) + CLIENT_PRIVATE(assigned) = +2
        """
        # USER sees public documents and their own private documents.
        f_user = build_qdrant_filter({"id": 1, "kind": "internal", "role": "user"}, [])
        assert isinstance(f_user.should, list)
        assert len(f_user.should) == 3

        # ADMIN (search): public + private(self) + group(bypass) + private(bypass) = 4
        f_admin = build_qdrant_filter({"id": 1, "kind": "internal", "role": "admin"}, [])
        assert isinstance(f_admin.should, list)
        assert len(f_admin.should) == 5

        # CURATOR no managed: public + private(self) = 2
        f_curator = build_qdrant_filter({"id": 1, "kind": "internal", "role": "curator"}, [])
        assert isinstance(f_curator.should, list)
        assert len(f_curator.should) == 3

        # CURATOR with managed: public + private(self) + private(assigned) + client(assigned) = 4
        f_curator_managed = build_qdrant_filter(
            {"id": 1, "kind": "internal", "role": "curator"},
            [],
            managed_client_ids=[100],
            managed_internal_ids=[200],
        )
        assert isinstance(f_curator_managed.should, list)
        assert len(f_curator_managed.should) == 5

    def test_visibility_conditions_vs_filter_always_consistent(self) -> None:
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
            assert isinstance(f.should, list)
            assert len(conds) == len(f.should), (
                f"Condition count mismatch for {kind}/{role}: "
                f"visibility_conditions={len(conds)}, filter={len(f.should)}"
            )


@pytest.mark.parametrize(
    "ctx",
    [
        internal_user(),
        internal_user(role=UserRole.ADMIN),
        internal_user(role=UserRole.CURATOR),
        client_user(),
    ],
    ids=["user", "admin", "curator", "client"],
)
@pytest.mark.parametrize("visibility", list(DocumentVisibility))
@pytest.mark.parametrize("for_list", [False, True], ids=["search", "listing"])
def test_client_public_and_denied_documents_across_roles(
    ctx: UserContext, visibility: DocumentVisibility, for_list: bool
) -> None:
    doc = Document(visibility=visibility, owner_id=999, group_id=999)
    expected = visibility == DocumentVisibility.CLIENT_PUBLIC or (
        ctx.user_kind == UserKind.INTERNAL
        and (
            visibility == DocumentVisibility.INTERNAL_PUBLIC
            or (ctx.is_admin and (for_list or visibility != DocumentVisibility.CLIENT_PRIVATE))
        )
    )
    actual = can_view_document(doc, ctx) if for_list else is_in_search_scope(doc, ctx)
    assert actual is expected

    table = Table(
        "documents",
        MetaData(),
        Column("id", Integer, primary_key=True),
        Column("visibility", String),
        Column("owner_id", Integer),
        Column("group_id", Integer),
    )
    engine = create_engine("sqlite://")
    table.metadata.create_all(engine)
    conditions = get_visibility_conditions(
        ctx.user_kind, list(ctx.group_ids), for_list=for_list, user_role=ctx.user_role
    )
    clauses = render_visibility_conditions(
        conditions,
        visibility_column=cast("InstrumentedAttribute[str]", table.c.visibility),
        owner_column=cast("InstrumentedAttribute[int | None]", table.c.owner_id),
        group_column=cast("InstrumentedAttribute[int | None]", table.c.group_id),
        user_id=ctx.user_id,
        group_ids=list(ctx.group_ids),
    )
    with engine.begin() as db:
        db.execute(table.insert().values(id=1, visibility=visibility.value, owner_id=999, group_id=999))
        assert (db.execute(select(table.c.id).where(or_(*clauses))).scalar_one_or_none() == 1) is expected
    engine.dispose()

    if not for_list:
        scope = build_qdrant_filter(ctx, list(ctx.group_ids))
        payload = {"metadata": {"visibility": visibility.value, "owner_id": 999, "group_id": 999}}
        assert check_filter(scope, payload, 1) is expected


@pytest.mark.parametrize(
    "ctx,allowed",
    [
        (internal_user(), False),
        (internal_user(role=UserRole.ADMIN), True),
        (internal_user(role=UserRole.CURATOR), True),
        (client_user(), False),
    ],
    ids=["user", "admin", "curator", "client"],
)
def test_client_public_requires_upload_for_clients_capability(ctx: UserContext, allowed: bool) -> None:
    if allowed:
        validate_document_visibility(DocumentVisibility.CLIENT_PUBLIC, None, ctx)
    else:
        with pytest.raises((BusinessRuleViolationError, ValidationError)):
            validate_document_visibility(DocumentVisibility.CLIENT_PUBLIC, None, ctx)


def test_clients_never_read_internal_documents_even_with_owner_and_group_match() -> None:
    ctx = client_user(group_ids=[5])

    for visibility in (
        DocumentVisibility.INTERNAL_PUBLIC,
        DocumentVisibility.INTERNAL_GROUP,
        DocumentVisibility.INTERNAL_PRIVATE,
    ):
        doc = Document(visibility=visibility, owner_id=ctx.user_id, group_id=5)
        assert not can_view_document(doc, ctx)
        assert not is_in_search_scope(doc, ctx)
