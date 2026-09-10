"""Tests for CURATOR role — access control, visibility, and scope management.

ТЗ §10 test scenarios:
1. Curator without assignments sees the same as regular user (fallback)
2. Curator sees client_private of assigned clients only
3. Curator sees internal_group docs of assigned groups (even without membership)
4. Curator cannot call PUT /admin/config (403)
5. Curator can manage docs in scope, not outside
6. Admin sees all docs in list (in_search_scope=false), not in chat search
7. Client sees only own client_private (regression)
8. Downgrading curator → user clears assignments
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from domain.entities.document import Document
from domain.exceptions import BusinessRuleViolation
from domain.services import can_view_document, get_visibility_conditions, is_in_search_scope
from domain.value_objects.owner_match import OwnerMatch
from domain.value_objects.roles import UserKind, UserRole
from domain.value_objects.user_context import UserContext
from domain.value_objects.visibility import DocumentVisibility
from infrastructure.acl import build_qdrant_filter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _curator(
    user_id=1, group_ids=None, managed_client_ids=None, managed_internal_ids=None, managed_group_ids=None
):
    return UserContext(
        user_id=user_id,
        user_kind=UserKind.INTERNAL,
        user_role=UserRole.CURATOR,
        group_ids=group_ids or [],
        managed_client_ids=managed_client_ids or [],
        managed_internal_ids=managed_internal_ids or [],
        managed_group_ids=managed_group_ids or [],
    )


def _internal_user(user_id=1, role=UserRole.USER, group_ids=None):
    return UserContext(
        user_id=user_id,
        user_kind=UserKind.INTERNAL,
        user_role=role,
        group_ids=group_ids or [],
    )


def _client_user(user_id=100):
    return UserContext(
        user_id=user_id,
        user_kind=UserKind.CLIENT,
        user_role=UserRole.USER,
    )


# ===========================================================================
# ТЗ §10.1: Curator without assignments sees same as regular user
# ===========================================================================


class TestCuratorFallback:
    def test_curator_without_assignments_sees_public(self):
        doc = SimpleNamespace(visibility="internal_public", owner_id=None, group_id=None)
        ctx = _curator()
        assert can_view_document(doc, ctx) is True

    def test_curator_without_assignments_sees_own_private(self):
        doc = SimpleNamespace(visibility="internal_private", owner_id=1, group_id=None)
        ctx = _curator(user_id=1)
        assert can_view_document(doc, ctx) is True

    def test_curator_without_assignments_not_others_private(self):
        doc = SimpleNamespace(visibility="internal_private", owner_id=99, group_id=None)
        ctx = _curator(user_id=1)
        assert can_view_document(doc, ctx) is False

    def test_curator_without_assignments_not_client_private(self):
        doc = SimpleNamespace(visibility="client_private", owner_id=100, group_id=None)
        ctx = _curator()
        assert can_view_document(doc, ctx) is False

    def test_curator_visibility_conditions_same_as_user(self):
        curator_conds = get_visibility_conditions(
            UserKind.INTERNAL,
            1,
            [],
            user_role=UserRole.CURATOR,
        )
        user_conds = get_visibility_conditions(
            UserKind.INTERNAL,
            1,
            [],
            user_role=UserRole.USER,
        )
        # Same number of conditions (public + private)
        assert len(curator_conds) == len(user_conds)


# ===========================================================================
# ТЗ §10.2: Curator sees assigned clients' private docs
# ===========================================================================


class TestCuratorAssignedClients:
    def test_sees_assigned_client_private(self):
        doc = SimpleNamespace(visibility="client_private", owner_id=100, group_id=None)
        ctx = _curator(managed_client_ids=[100])
        assert can_view_document(doc, ctx) is True

    def test_not_unassigned_client_private(self):
        doc = SimpleNamespace(visibility="client_private", owner_id=200, group_id=None)
        ctx = _curator(managed_client_ids=[100])
        assert can_view_document(doc, ctx) is False

    def test_sees_only_assigned_multiple_clients(self):
        doc_100 = SimpleNamespace(visibility="client_private", owner_id=100, group_id=None)
        doc_200 = SimpleNamespace(visibility="client_private", owner_id=200, group_id=None)
        doc_300 = SimpleNamespace(visibility="client_private", owner_id=300, group_id=None)
        ctx = _curator(managed_client_ids=[100, 200])
        assert can_view_document(doc_100, ctx) is True
        assert can_view_document(doc_200, ctx) is True
        assert can_view_document(doc_300, ctx) is False


# ===========================================================================
# ТЗ §10.3: Curator sees assigned group docs
# ===========================================================================


class TestCuratorAssignedGroups:
    def test_sees_assigned_group_docs(self):
        doc = SimpleNamespace(visibility="internal_group", owner_id=None, group_id=5)
        ctx = _curator(managed_group_ids=[5])
        assert can_view_document(doc, ctx) is True

    def test_not_unassigned_group_docs(self):
        doc = SimpleNamespace(visibility="internal_group", owner_id=None, group_id=99)
        ctx = _curator(managed_group_ids=[5])
        assert can_view_document(doc, ctx) is False

    def test_sees_own_membership_group_docs(self):
        doc = SimpleNamespace(visibility="internal_group", owner_id=None, group_id=10)
        ctx = _curator(group_ids=[10])
        assert can_view_document(doc, ctx) is True

    def test_sees_union_of_own_and_managed_groups(self):
        doc_own = SimpleNamespace(visibility="internal_group", owner_id=None, group_id=10)
        doc_managed = SimpleNamespace(visibility="internal_group", owner_id=None, group_id=20)
        doc_none = SimpleNamespace(visibility="internal_group", owner_id=None, group_id=99)
        ctx = _curator(group_ids=[10], managed_group_ids=[20])
        assert can_view_document(doc_own, ctx) is True
        assert can_view_document(doc_managed, ctx) is True
        assert can_view_document(doc_none, ctx) is False


# ===========================================================================
# ТЗ §10.5: Curator manages docs in scope, not outside
# ===========================================================================


class TestCuratorDocumentPermissions:
    def test_can_edit_assigned_client_doc(self):
        doc = Document(
            visibility=DocumentVisibility.CLIENT_PRIVATE,
            owner_id=100,
        )
        ctx = _curator(managed_client_ids=[100])
        assert (
            doc.can_edit_chunks(
                ctx.user_id,
                UserRole.CURATOR,
                ctx.group_ids,
                managed_client_ids=ctx.managed_client_ids,
            )
            is True
        )

    def test_cannot_edit_unassigned_client_doc(self):
        doc = Document(
            visibility=DocumentVisibility.CLIENT_PRIVATE,
            owner_id=200,
        )
        ctx = _curator(managed_client_ids=[100])
        assert (
            doc.can_edit_chunks(
                ctx.user_id,
                UserRole.CURATOR,
                ctx.group_ids,
                managed_client_ids=ctx.managed_client_ids,
            )
            is False
        )

    def test_can_delete_assigned_internal_doc(self):
        doc = Document(
            visibility=DocumentVisibility.INTERNAL_PRIVATE,
            owner_id=50,
        )
        ctx = _curator(managed_internal_ids=[50])
        assert (
            doc.can_be_deleted_by(
                ctx.user_id,
                UserRole.CURATOR,
                ctx.group_ids,
                managed_internal_ids=ctx.managed_internal_ids,
            )
            is True
        )

    def test_cannot_delete_unassigned_internal_doc(self):
        doc = Document(
            visibility=DocumentVisibility.INTERNAL_PRIVATE,
            owner_id=99,
        )
        ctx = _curator(managed_internal_ids=[50])
        assert (
            doc.can_be_deleted_by(
                ctx.user_id,
                UserRole.CURATOR,
                ctx.group_ids,
                managed_internal_ids=ctx.managed_internal_ids,
            )
            is False
        )

    def test_can_edit_managed_group_doc(self):
        doc = Document(
            visibility=DocumentVisibility.INTERNAL_GROUP,
            group_id=5,
        )
        ctx = _curator(managed_group_ids=[5])
        assert (
            doc.can_edit_chunks(
                ctx.user_id,
                UserRole.CURATOR,
                ctx.group_ids,
                managed_group_ids=ctx.managed_group_ids,
            )
            is True
        )


# ===========================================================================
# ТЗ §10.6: Admin list vs search scope separation
# ===========================================================================


class TestAdminListVsSearch:
    def test_admin_sees_client_private_in_list(self):
        doc = SimpleNamespace(visibility="client_private", owner_id=100, group_id=None)
        ctx = _internal_user(role=UserRole.ADMIN)
        assert can_view_document(doc, ctx) is True

    def test_admin_client_private_not_in_search(self):
        doc = SimpleNamespace(visibility="client_private", owner_id=100, group_id=None)
        ctx = _internal_user(role=UserRole.ADMIN)
        assert is_in_search_scope(doc, ctx) is False


# ===========================================================================
# ТЗ §10.7: Client regression test
# ===========================================================================


class TestClientRegression:
    def test_client_sees_only_own_private(self):
        doc = SimpleNamespace(visibility="client_private", owner_id=100, group_id=None)
        ctx = _client_user(user_id=100)
        assert can_view_document(doc, ctx) is True

    def test_client_not_others_private(self):
        doc = SimpleNamespace(visibility="client_private", owner_id=200, group_id=None)
        ctx = _client_user(user_id=100)
        assert can_view_document(doc, ctx) is False

    def test_client_not_internal_public(self):
        doc = SimpleNamespace(visibility="internal_public", owner_id=None, group_id=None)
        ctx = _client_user(user_id=100)
        assert can_view_document(doc, ctx) is False

    def test_client_not_internal_group(self):
        doc = SimpleNamespace(visibility="internal_group", owner_id=None, group_id=5)
        ctx = _client_user(user_id=100)
        assert can_view_document(doc, ctx) is False


# ===========================================================================
# OwnerMatch.ASSIGNED tests
# ===========================================================================


class TestOwnerMatchAssigned:
    def test_assigned_match_with_owner_in_list(self):
        conds = get_visibility_conditions(
            UserKind.INTERNAL,
            1,
            [],
            user_role=UserRole.CURATOR,
            managed_client_ids=[100],
        )
        # Find the client_private condition
        client_cond = [c for c in conds if c.visibility == DocumentVisibility.CLIENT_PRIVATE]
        assert len(client_cond) == 1
        assert client_cond[0].owner_match == OwnerMatch.ASSIGNED
        assert 100 in client_cond[0].owner_ids

    def test_assigned_qdrant_filter(self):
        user = {"id": 1, "kind": "internal", "role": "curator"}
        f = build_qdrant_filter(user, [], managed_client_ids=[100])
        # Should have conditions for public, private(own), client_private(assigned)
        assert f.should is not None
        assert len(f.should) >= 3


# ===========================================================================
# Capabilities tests
# ===========================================================================


class TestCuratorCapabilities:
    def test_curator_has_documents_manage(self):
        from domain.value_objects.capabilities import Capability, get_role_capabilities

        caps = get_role_capabilities(UserRole.CURATOR)
        assert Capability.DOCUMENTS_MANAGE in caps

    def test_curator_has_chunks_manage(self):
        from domain.value_objects.capabilities import Capability, get_role_capabilities

        caps = get_role_capabilities(UserRole.CURATOR)
        assert Capability.CHUNKS_MANAGE in caps

    def test_curator_no_config_manage(self):
        from domain.value_objects.capabilities import Capability, get_role_capabilities

        caps = get_role_capabilities(UserRole.CURATOR)
        assert Capability.CONFIG_MANAGE not in caps

    def test_curator_no_users_manage(self):
        from domain.value_objects.capabilities import Capability, get_role_capabilities

        caps = get_role_capabilities(UserRole.CURATOR)
        assert Capability.USERS_MANAGE not in caps

    def test_user_has_documents_view(self):
        from domain.value_objects.capabilities import Capability, get_role_capabilities

        caps = get_role_capabilities(UserRole.USER)
        assert Capability.DOCUMENTS_VIEW in caps
        assert Capability.DOCUMENTS_MANAGE not in caps


# ===========================================================================
# User entity role validation
# ===========================================================================


class TestUserRoleValidation:
    def test_client_cannot_become_curator(self):
        from domain.entities.user import User

        user = User(id=1, kind=UserKind.CLIENT, role=UserRole.USER)
        with pytest.raises(BusinessRuleViolation, match="cannot be admin or curator"):
            user.change_role(UserRole.CURATOR)

    def test_user_can_become_curator(self):
        from domain.entities.user import User

        user = User(id=1, kind=UserKind.INTERNAL, role=UserRole.USER)
        user.change_role(UserRole.CURATOR)
        assert user.role == UserRole.CURATOR

    def test_curator_can_become_user(self):
        from domain.entities.user import User

        user = User(id=1, kind=UserKind.INTERNAL, role=UserRole.CURATOR)
        user.change_role(UserRole.USER)
        assert user.role == UserRole.USER
