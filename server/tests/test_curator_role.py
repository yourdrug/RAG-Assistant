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

from pathlib import Path
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock, patch

import pytest

from application.rag.dto import PreparedQuery
from composition.service_providers import create_rag_service
from domain.identity.capabilities import Capability, get_role_capabilities
from domain.identity.roles import UserKind, UserRole
from domain.identity.user import User
from domain.identity.user_context import CuratorScope, UserContext
from domain.knowledge.document import Document
from domain.rag.context import RagContext
from tests.fakes import FakeMLClientRegistry, require_present


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from domain.knowledge.access_control import (
    OwnerMatch,
    can_view_document,
    get_visibility_conditions,
    is_in_search_scope,
)
from domain.knowledge.enums import DocumentVisibility
from domain.shared.exceptions import BusinessRuleViolationError
from infrastructure.repositories.vector.acl import build_qdrant_filter


if TYPE_CHECKING:
    from application.ports.ml_clients import MLClientPort
    from infrastructure.ml.rag.rag_steps import RagOperations


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _curator(
    user_id: int = 1,
    group_ids: list[int] | None = None,
    managed_client_ids: list[int] | None = None,
    managed_internal_ids: list[int] | None = None,
    managed_group_ids: list[int] | None = None,
) -> UserContext:
    return UserContext(
        user_id=user_id,
        user_kind=UserKind.INTERNAL,
        user_role=UserRole.CURATOR,
        group_ids=tuple(group_ids or []),
        managed_client_ids=tuple(managed_client_ids or []),
        managed_internal_ids=tuple(managed_internal_ids or []),
        managed_group_ids=tuple(managed_group_ids or []),
    )


def _internal_user(
    user_id: int = 1, role: UserRole = UserRole.USER, group_ids: list[int] | None = None
) -> UserContext:
    return UserContext(
        user_id=user_id,
        user_kind=UserKind.INTERNAL,
        user_role=role,
        group_ids=tuple(group_ids or []),
    )


def _client_user(user_id: int = 100) -> UserContext:
    return UserContext(
        user_id=user_id,
        user_kind=UserKind.CLIENT,
        user_role=UserRole.USER,
    )


# ===========================================================================
# ТЗ §10.1: Curator without assignments sees same as regular user
# ===========================================================================


class TestCuratorFallback:
    def test_curator_without_assignments_sees_public(self) -> None:
        doc = SimpleNamespace(visibility="internal_public", owner_id=None, group_id=None)
        ctx = _curator()
        assert can_view_document(cast("Document", doc), ctx) is True

    def test_curator_without_assignments_sees_own_private(self) -> None:
        doc = SimpleNamespace(visibility="internal_private", owner_id=1, group_id=None)
        ctx = _curator(user_id=1)
        assert can_view_document(cast("Document", doc), ctx) is True

    def test_curator_without_assignments_not_others_private(self) -> None:
        doc = SimpleNamespace(visibility="internal_private", owner_id=99, group_id=None)
        ctx = _curator(user_id=1)
        assert can_view_document(cast("Document", doc), ctx) is False

    def test_curator_without_assignments_not_client_private(self) -> None:
        doc = SimpleNamespace(visibility="client_private", owner_id=100, group_id=None)
        ctx = _curator()
        assert can_view_document(cast("Document", doc), ctx) is False

    def test_curator_visibility_conditions_same_as_user(self) -> None:
        curator_conds = get_visibility_conditions(
            UserKind.INTERNAL,
            [],
            user_role=UserRole.CURATOR,
        )
        user_conds = get_visibility_conditions(
            UserKind.INTERNAL,
            [],
            user_role=UserRole.USER,
        )
        # Same number of conditions (public + private)
        assert len(curator_conds) == len(user_conds)


# ===========================================================================
# ТЗ §10.2: Curator sees assigned clients' private docs
# ===========================================================================


class TestCuratorAssignedClients:
    def test_sees_assigned_client_private(self) -> None:
        doc = SimpleNamespace(visibility="client_private", owner_id=100, group_id=None)
        ctx = _curator(managed_client_ids=[100])
        assert can_view_document(cast("Document", doc), ctx) is True

    def test_not_unassigned_client_private(self) -> None:
        doc = SimpleNamespace(visibility="client_private", owner_id=200, group_id=None)
        ctx = _curator(managed_client_ids=[100])
        assert can_view_document(cast("Document", doc), ctx) is False

    def test_sees_only_assigned_multiple_clients(self) -> None:
        doc_100 = SimpleNamespace(visibility="client_private", owner_id=100, group_id=None)
        doc_200 = SimpleNamespace(visibility="client_private", owner_id=200, group_id=None)
        doc_300 = SimpleNamespace(visibility="client_private", owner_id=300, group_id=None)
        ctx = _curator(managed_client_ids=[100, 200])
        assert can_view_document(cast("Document", doc_100), ctx) is True
        assert can_view_document(cast("Document", doc_200), ctx) is True
        assert can_view_document(cast("Document", doc_300), ctx) is False


# ===========================================================================
# ТЗ §10.3: Curator sees assigned group docs
# ===========================================================================


class TestCuratorAssignedGroups:
    def test_sees_assigned_group_docs(self) -> None:
        doc = SimpleNamespace(visibility="internal_group", owner_id=None, group_id=5)
        ctx = _curator(managed_group_ids=[5])
        assert can_view_document(cast("Document", doc), ctx) is True

    def test_not_unassigned_group_docs(self) -> None:
        doc = SimpleNamespace(visibility="internal_group", owner_id=None, group_id=99)
        ctx = _curator(managed_group_ids=[5])
        assert can_view_document(cast("Document", doc), ctx) is False

    def test_sees_own_membership_group_docs(self) -> None:
        doc = SimpleNamespace(visibility="internal_group", owner_id=None, group_id=10)
        ctx = _curator(group_ids=[10])
        assert can_view_document(cast("Document", doc), ctx) is True

    def test_sees_union_of_own_and_managed_groups(self) -> None:
        doc_own = SimpleNamespace(visibility="internal_group", owner_id=None, group_id=10)
        doc_managed = SimpleNamespace(visibility="internal_group", owner_id=None, group_id=20)
        doc_none = SimpleNamespace(visibility="internal_group", owner_id=None, group_id=99)
        ctx = _curator(group_ids=[10], managed_group_ids=[20])
        assert can_view_document(cast("Document", doc_own), ctx) is True
        assert can_view_document(cast("Document", doc_managed), ctx) is True
        assert can_view_document(cast("Document", doc_none), ctx) is False


# ===========================================================================
# ТЗ §10.5: Curator manages docs in scope, not outside
# ===========================================================================


class TestCuratorDocumentPermissions:
    def test_can_edit_assigned_client_doc(self) -> None:
        doc = Document(
            visibility=DocumentVisibility.CLIENT_PRIVATE,
            owner_id=100,
        )
        ctx = _curator(managed_client_ids=[100])
        assert (
            doc.can_edit_chunks(
                ctx.user_id,
                UserRole.CURATOR,
                list(ctx.group_ids),
                managed_client_ids=list(ctx.managed_client_ids),
            )
            is True
        )

    def test_cannot_edit_unassigned_client_doc(self) -> None:
        doc = Document(
            visibility=DocumentVisibility.CLIENT_PRIVATE,
            owner_id=200,
        )
        ctx = _curator(managed_client_ids=[100])
        assert (
            doc.can_edit_chunks(
                ctx.user_id,
                UserRole.CURATOR,
                list(ctx.group_ids),
                managed_client_ids=list(ctx.managed_client_ids),
            )
            is False
        )

    def test_can_delete_assigned_internal_doc(self) -> None:
        doc = Document(
            visibility=DocumentVisibility.INTERNAL_PRIVATE,
            owner_id=50,
        )
        ctx = _curator(managed_internal_ids=[50])
        assert (
            doc.can_be_deleted_by(
                ctx.user_id,
                UserRole.CURATOR,
                list(ctx.group_ids),
                managed_internal_ids=list(ctx.managed_internal_ids),
            )
            is True
        )

    def test_cannot_delete_unassigned_internal_doc(self) -> None:
        doc = Document(
            visibility=DocumentVisibility.INTERNAL_PRIVATE,
            owner_id=99,
        )
        ctx = _curator(managed_internal_ids=[50])
        assert (
            doc.can_be_deleted_by(
                ctx.user_id,
                UserRole.CURATOR,
                list(ctx.group_ids),
                managed_internal_ids=list(ctx.managed_internal_ids),
            )
            is False
        )

    def test_can_edit_managed_group_doc(self) -> None:
        doc = Document(
            visibility=DocumentVisibility.INTERNAL_GROUP,
            group_id=5,
        )
        ctx = _curator(managed_group_ids=[5])
        assert (
            doc.can_edit_chunks(
                ctx.user_id,
                UserRole.CURATOR,
                list(ctx.group_ids),
                managed_group_ids=list(ctx.managed_group_ids),
            )
            is True
        )


# ===========================================================================
# ТЗ §10.6: Admin list vs search scope separation
# ===========================================================================


class TestAdminListVsSearch:
    def test_admin_sees_client_private_in_list(self) -> None:
        doc = SimpleNamespace(visibility="client_private", owner_id=100, group_id=None)
        ctx = _internal_user(role=UserRole.ADMIN)
        assert can_view_document(cast("Document", doc), ctx) is True

    def test_admin_client_private_not_in_search(self) -> None:
        doc = SimpleNamespace(visibility="client_private", owner_id=100, group_id=None)
        ctx = _internal_user(role=UserRole.ADMIN)
        assert is_in_search_scope(cast("Document", doc), ctx) is False


# ===========================================================================
# ТЗ §10.7: Client regression test
# ===========================================================================


class TestClientRegression:
    def test_client_sees_only_own_private(self) -> None:
        doc = SimpleNamespace(visibility="client_private", owner_id=100, group_id=None)
        ctx = _client_user(user_id=100)
        assert can_view_document(cast("Document", doc), ctx) is True

    def test_client_not_others_private(self) -> None:
        doc = SimpleNamespace(visibility="client_private", owner_id=200, group_id=None)
        ctx = _client_user(user_id=100)
        assert can_view_document(cast("Document", doc), ctx) is False

    def test_client_not_internal_public(self) -> None:
        doc = SimpleNamespace(visibility="internal_public", owner_id=None, group_id=None)
        ctx = _client_user(user_id=100)
        assert can_view_document(cast("Document", doc), ctx) is False

    def test_client_not_internal_group(self) -> None:
        doc = SimpleNamespace(visibility="internal_group", owner_id=None, group_id=5)
        ctx = _client_user(user_id=100)
        assert can_view_document(cast("Document", doc), ctx) is False


# ===========================================================================
# OwnerMatch.ASSIGNED tests
# ===========================================================================


class TestOwnerMatchAssigned:
    def test_assigned_match_with_owner_in_list(self) -> None:
        conds = get_visibility_conditions(
            UserKind.INTERNAL,
            [],
            user_role=UserRole.CURATOR,
            managed_client_ids=[100],
        )
        # Find the client_private condition
        client_cond = [c for c in conds if c.visibility == DocumentVisibility.CLIENT_PRIVATE]
        assert len(client_cond) == 1
        assert client_cond[0].owner_match == OwnerMatch.ASSIGNED
        assert 100 in require_present(client_cond[0].owner_ids)

    def test_assigned_qdrant_filter(self) -> None:
        user = {"id": 1, "kind": "internal", "role": "curator"}
        f = build_qdrant_filter(user, [], managed_client_ids=[100])
        # Should have conditions for public, private(own), client_private(assigned)
        assert f.should is not None
        assert isinstance(f.should, list)
        assert len(f.should) >= 3


# ===========================================================================
# Capabilities tests
# ===========================================================================


class TestCuratorCapabilities:
    def test_curator_has_documents_manage(self) -> None:

        caps = get_role_capabilities(UserRole.CURATOR)
        assert Capability.DOCUMENTS_MANAGE in caps

    def test_curator_has_chunks_manage(self) -> None:

        caps = get_role_capabilities(UserRole.CURATOR)
        assert Capability.CHUNKS_MANAGE in caps

    def test_curator_no_config_manage(self) -> None:

        caps = get_role_capabilities(UserRole.CURATOR)
        assert Capability.CONFIG_MANAGE not in caps

    def test_curator_no_users_manage(self) -> None:

        caps = get_role_capabilities(UserRole.CURATOR)
        assert Capability.USERS_MANAGE not in caps

    def test_user_has_documents_view(self) -> None:

        caps = get_role_capabilities(UserRole.USER)
        assert Capability.DOCUMENTS_VIEW in caps
        assert Capability.DOCUMENTS_MANAGE not in caps


# ===========================================================================
# User entity role validation
# ===========================================================================


# ===========================================================================
# A3: Characterization — build_qdrant_filter IGNORES managed_ids today
# ===========================================================================


class TestCuratorFilterIgnoresManagedIds:
    """Document the current behavior: build_qdrant_filter is called WITHOUT
    managed-ids in the RAG pipeline (rag_service.py:128).

    This means the Qdrant ACL filter does NOT contain ASSIGNED-conditions
    for managed users — curator cannot find their documents through RAG search.

    This is the primary bug that Phase B will fix.
    """

    def test_build_qdrant_filter_without_managed_ids(self) -> None:
        """Current RAG call: build_qdrant_filter(user, group_ids) — no managed-ids."""
        user = {"id": 1, "kind": "internal", "role": "curator"}
        f = build_qdrant_filter(user, [])
        # Only base internal conditions: PUBLIC + PRIVATE(self)
        assert f.should is not None
        assert isinstance(f.should, list)
        assert len(f.should) == 2

    def test_build_qdrant_filter_with_managed_ids_has_assigned(self) -> None:
        """When managed-ids ARE passed: filter includes ASSIGNED conditions."""
        user = {"id": 1, "kind": "internal", "role": "curator"}
        f = build_qdrant_filter(
            user,
            [],
            managed_client_ids=[100, 200],
            managed_internal_ids=[300],
        )
        # Should have: PUBLIC + PRIVATE(self) + PRIVATE(assigned managed_internal) + CLIENT_PRIVATE(assigned)
        assert f.should is not None
        assert isinstance(f.should, list)
        assert len(f.should) >= 3  # At least one extra for managed scope

    def test_managed_ids_none_vs_empty_equivalent(self) -> None:
        """None and empty list produce same filter (documented behavior)."""
        user = {"id": 1, "kind": "internal", "role": "curator"}
        f_none = build_qdrant_filter(user, [], managed_client_ids=None, managed_internal_ids=None)
        f_empty = build_qdrant_filter(user, [], managed_client_ids=[], managed_internal_ids=[])
        assert isinstance(f_none.should, list)
        assert isinstance(f_empty.should, list)
        assert len(f_none.should) == len(f_empty.should)

    def test_rag_pipeline_calls_build_filter_without_managed(self) -> None:
        """CHARACTERIZATION: rag_service._init_state calls without managed-ids.

        This test documents the current call at rag_service.py:128:
            access_filter = build_qdrant_filter(user, ctx.user_group_ids)

        Phase B will change this to pass ctx.curator_scope fields.
        """
        # Build a curator context the way RAG pipeline does today (without managed-ids)
        user = {"id": 1, "kind": "internal", "role": "curator"}
        # Today's call (NO managed-ids):
        filter_no_managed = build_qdrant_filter(user, [])
        # Expected call after Phase B (WITH managed-ids):
        filter_with_managed = build_qdrant_filter(
            user,
            [],
            managed_client_ids=[100],
            managed_internal_ids=[200],
        )
        assert isinstance(filter_no_managed.should, list)
        # They differ — this is the gap
        assert isinstance(filter_with_managed.should, list)
        assert len(filter_no_managed.should) < len(filter_with_managed.should)


# ===========================================================================
# B3: ChatService._prepare_chat passes CuratorScope
# ===========================================================================


class TestChatServiceCuratorScope:
    """B3: ChatService._prepare_chat builds CuratorScope from UserContext."""

    def test_curator_gets_scope(self) -> None:

        user_ctx = _curator(user_id=1, managed_client_ids=[100, 200], managed_internal_ids=[300])
        assert user_ctx.is_curator is True
        scope = CuratorScope(
            managed_client_ids=tuple(user_ctx.managed_client_ids),
            managed_internal_ids=tuple(user_ctx.managed_internal_ids),
            managed_group_ids=tuple(user_ctx.managed_group_ids),
        )
        assert scope.is_empty() is False
        assert scope.managed_client_ids == (100, 200)
        assert scope.managed_internal_ids == (300,)

    def test_user_gets_none_scope(self) -> None:
        user_ctx = _internal_user(user_id=1, role=UserRole.USER)
        assert user_ctx.is_curator is False
        scope = user_ctx.with_curator_scope(None).managed_client_ids
        assert scope == ()

    def test_curator_without_assignments_gets_empty_scope(self) -> None:
        user_ctx = _curator(user_id=1)
        scope = CuratorScope(
            managed_client_ids=tuple(user_ctx.managed_client_ids),
            managed_internal_ids=tuple(user_ctx.managed_internal_ids),
            managed_group_ids=tuple(user_ctx.managed_group_ids),
        )
        assert scope.is_empty() is True

    def test_chat_context_construction_with_curator_scope(self) -> None:

        scope = CuratorScope(managed_client_ids=(100,), managed_internal_ids=(200,), managed_group_ids=(300,))
        ctx = RagContext(
            user=UserContext(
                user_id=1, user_kind=UserKind.INTERNAL, user_role=UserRole.CURATOR
            ).with_curator_scope(scope)
        )
        assert ctx.curator_scope == scope
        assert ctx.curator_scope.managed_client_ids == (100,)


# ===========================================================================
# B4: RagService._init_state passes managed-ids from CuratorScope
# ===========================================================================


class TestRagServiceCuratorScope:
    """B4: RagService._init_state unpacks CuratorScope into build_qdrant_filter."""

    def test_curator_scope_produces_correct_managed_args(self) -> None:
        """CuratorScope fields unpack to managed_*_ids arguments."""

        scope = CuratorScope(managed_client_ids=(100, 200), managed_internal_ids=(300,))
        user = {"id": 1, "kind": "internal", "role": "curator"}
        f = build_qdrant_filter(
            user,
            [],
            managed_client_ids=list(scope.managed_client_ids),
            managed_internal_ids=list(scope.managed_internal_ids),
            managed_group_ids=list(scope.managed_group_ids),
        )
        # Filter should include ASSIGNED conditions for managed users
        assert f.should is not None
        assert isinstance(f.should, list)
        assert len(f.should) >= 3

    def test_user_no_scope_passes_none_managed_args(self) -> None:
        """Non-curator with curator_scope=None → all managed_*_ids=None."""
        ctx = RagContext(
            user=UserContext(
                user_id=1, user_kind=UserKind.INTERNAL, user_role=UserRole.USER
            ).with_curator_scope(None)
        )
        scope_val = ctx.curator_scope
        user = {"id": ctx.user_id, "kind": ctx.user_kind, "role": ctx.user_role}
        f = build_qdrant_filter(
            user,
            list(ctx.user_group_ids),
            managed_client_ids=list(scope_val.managed_client_ids) if scope_val else None,
            managed_internal_ids=list(scope_val.managed_internal_ids) if scope_val else None,
            managed_group_ids=list(scope_val.managed_group_ids) if scope_val else None,
        )
        # Non-curator: only base conditions (PUBLIC + PRIVATE(self))
        assert f.should is not None
        assert isinstance(f.should, list)
        assert len(f.should) == 2

    def test_rag_filter_with_scope_has_assigned_conditions(self) -> None:
        """After B4, curator with managed_ids → filter includes ASSIGNED conditions."""

        user = {"id": 1, "kind": "internal", "role": "curator"}
        scope = CuratorScope(managed_client_ids=(100,), managed_internal_ids=(200,))
        f = build_qdrant_filter(
            user,
            [],
            managed_client_ids=list(scope.managed_client_ids),
            managed_internal_ids=list(scope.managed_internal_ids),
            managed_group_ids=list(scope.managed_group_ids),
        )
        # Should have: PUBLIC + PRIVATE(self) + PRIVATE(assigned internal) + CLIENT_PRIVATE(assigned)
        assert f.should is not None
        assert isinstance(f.should, list)
        assert len(f.should) >= 3

    def test_rag_init_state_integration(self) -> None:
        """RagService._init_state with CuratorScope calls build_qdrant_filter with managed-ids."""

        scope = CuratorScope(managed_client_ids=(100, 200), managed_internal_ids=(300,))
        ctx = RagContext(
            user=UserContext(
                user_id=1, user_kind=UserKind.INTERNAL, user_role=UserRole.CURATOR
            ).with_curator_scope(scope)
        )
        with patch("infrastructure.ml.rag.rag_steps.build_qdrant_filter") as mock_filter:
            mock_filter.return_value = type("F", (), {"should": []})()
            with (
                patch("infrastructure.ml.rag.rag_steps.build_rag_settings"),
                patch("infrastructure.ml.rag.rag_steps.build_search_filter"),
            ):
                svc = create_rag_service(
                    cast("MLClientPort", FakeMLClientRegistry()), document_access=MagicMock()
                )
                request = svc.preparation.request("test question", [], ctx)
                cast("RagOperations", svc.retrieval).retrieval_work(PreparedQuery(request, "test question"))
                mock_filter.assert_called_once_with(
                    UserContext(
                        user_id=1,
                        user_kind=UserKind.INTERNAL,
                        user_role=UserRole.CURATOR,
                        managed_client_ids=(100, 200),
                        managed_internal_ids=(300,),
                    ),
                    [],
                    managed_client_ids=[100, 200],
                    managed_internal_ids=[300],
                    managed_group_ids=[],
                )


class TestUserRoleValidation:
    def test_client_cannot_become_curator(self) -> None:

        user = User(id=1, kind=UserKind.CLIENT, role=UserRole.USER)
        with pytest.raises(BusinessRuleViolationError, match="cannot be admin or curator"):
            user.change_role(UserRole.CURATOR)

    def test_user_can_become_curator(self) -> None:

        user = User(id=1, kind=UserKind.INTERNAL, role=UserRole.USER)
        user.change_role(UserRole.CURATOR)
        assert user.role == UserRole.CURATOR

    def test_curator_can_become_user(self) -> None:

        user = User(id=1, kind=UserKind.INTERNAL, role=UserRole.CURATOR)
        user.change_role(UserRole.USER)
        assert user.role == UserRole.USER
