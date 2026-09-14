"""Tests for CuratorScope value object and ChatContext curator_scope field.

Phase B of curator-scope-and-bm25-acl-implementation-plan.md:
- B1: CuratorScope VO (frozen, hashable, is_empty)
- B2: ChatContext.curator_scope default None
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from domain.value_objects.chat_context import ChatContext
from domain.value_objects.curator_scope import CuratorScope


# ===========================================================================
# B1: CuratorScope VO
# ===========================================================================


class TestCuratorScope:
    def test_frozen(self):
        scope = CuratorScope(managed_client_ids=(100,), managed_internal_ids=(200,))
        with pytest.raises(AttributeError):
            scope.managed_client_ids = (999,)

    def test_hashable(self):
        scope = CuratorScope(managed_client_ids=(100,), managed_internal_ids=(200,))
        assert isinstance(hash(scope), int)

    def test_hash_equal_for_same_data(self):
        a = CuratorScope(managed_client_ids=(100,), managed_internal_ids=(200,))
        b = CuratorScope(managed_client_ids=(100,), managed_internal_ids=(200,))
        assert hash(a) == hash(b)
        assert a == b

    def test_hash_differs_for_different_data(self):
        a = CuratorScope(managed_client_ids=(100,))
        b = CuratorScope(managed_client_ids=(200,))
        assert a != b

    def test_default_empty_tuples(self):
        scope = CuratorScope()
        assert scope.managed_client_ids == ()
        assert scope.managed_internal_ids == ()
        assert scope.managed_group_ids == ()

    def test_is_empty_true_for_default(self):
        assert CuratorScope().is_empty() is True

    def test_is_empty_false_when_client_ids(self):
        assert CuratorScope(managed_client_ids=(100,)).is_empty() is False

    def test_is_empty_false_when_internal_ids(self):
        assert CuratorScope(managed_internal_ids=(200,)).is_empty() is False

    def test_is_empty_false_when_group_ids(self):
        assert CuratorScope(managed_group_ids=(300,)).is_empty() is False

    def test_tuples_not_lists(self):
        scope = CuratorScope(managed_client_ids=(1, 2), managed_internal_ids=(3,))
        assert isinstance(scope.managed_client_ids, tuple)
        assert isinstance(scope.managed_internal_ids, tuple)


# ===========================================================================
# B2: ChatContext.curator_scope default
# ===========================================================================


class TestChatContextCuratorScope:
    def test_default_none(self):
        ctx = ChatContext(user_id=1, user_kind="internal")
        assert ctx.curator_scope is None

    def test_explicit_none(self):
        ctx = ChatContext(user_id=1, user_kind="internal", curator_scope=None)
        assert ctx.curator_scope is None

    def test_explicit_scope(self):
        scope = CuratorScope(managed_client_ids=(100,))
        ctx = ChatContext(user_id=1, user_kind="internal", curator_scope=scope)
        assert ctx.curator_scope is scope
        assert ctx.curator_scope.managed_client_ids == (100,)

    def test_frozen_with_scope(self):
        scope = CuratorScope(managed_client_ids=(100,))
        ctx = ChatContext(user_id=1, user_kind="internal", curator_scope=scope)
        with pytest.raises(AttributeError):
            ctx.curator_scope = None

    def test_existing_constructors_unaffected(self):
        ctx = ChatContext(user_id=1, user_kind="internal", user_role="admin", user_group_ids=[10])
        assert ctx.curator_scope is None
        assert ctx.user_id == 1
        assert ctx.user_group_ids == [10]
