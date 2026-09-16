"""Tests for answer cache hash behavior — Phase C.

compute_visibility_scope_hash now includes user_role and curator_scope,
closing the demotion cache poisoning vulnerability.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from domain.value_objects.curator_scope import CuratorScope
from infrastructure.ml.answer_cache import CACHE_PREFIX, compute_visibility_scope_hash


# ---------------------------------------------------------------------------
# Cache prefix
# ---------------------------------------------------------------------------


class TestCachePrefix:
    def test_prefix_has_no_version(self):
        assert CACHE_PREFIX == "rag:cache:v3:"


# ---------------------------------------------------------------------------
# Phase C: hash includes role and curator_scope
# ---------------------------------------------------------------------------


class TestCacheHashIncludesRoleAndScope:
    """Cache hash now includes user_role and curator_scope."""

    def test_hash_includes_role(self):
        h_user = compute_visibility_scope_hash("internal", 1, [], user_role="user")
        h_curator = compute_visibility_scope_hash("internal", 1, [], user_role="curator")
        assert h_user != h_curator

    def test_hash_includes_curator_scope(self):
        scope = CuratorScope(managed_client_ids=(10, 20))
        h_with = compute_visibility_scope_hash("internal", 1, [], user_role="curator", curator_scope=scope)
        h_without = compute_visibility_scope_hash("internal", 1, [], user_role="curator")
        assert h_with != h_without

    def test_hash_no_scope_vs_none(self):
        h_none = compute_visibility_scope_hash("internal", 1, [], curator_scope=None)
        h_empty = compute_visibility_scope_hash("internal", 1, [], curator_scope=CuratorScope())
        assert h_none == h_empty

    def test_hash_empty_curator_scope(self):
        h1 = compute_visibility_scope_hash("internal", 1, [], user_role="curator")
        h2 = compute_visibility_scope_hash(
            "internal", 1, [], user_role="curator", curator_scope=CuratorScope()
        )
        assert h1 == h2

    def test_demotion_now_different_hash(self):
        scope = CuratorScope(managed_client_ids=(10, 20))
        h_before = compute_visibility_scope_hash(
            "internal", 42, [10, 20], user_role="curator", curator_scope=scope
        )
        h_after = compute_visibility_scope_hash("internal", 42, [10, 20], user_role="user")
        assert h_before != h_after

    def test_assignment_change_different_hash(self):
        scope_before = CuratorScope(managed_client_ids=(10,))
        scope_after = CuratorScope(managed_client_ids=(10, 20, 30))
        h1 = compute_visibility_scope_hash(
            "internal", 42, [], user_role="curator", curator_scope=scope_before
        )
        h2 = compute_visibility_scope_hash("internal", 42, [], user_role="curator", curator_scope=scope_after)
        assert h1 != h2


# ---------------------------------------------------------------------------
# Backward compat defaults
# ---------------------------------------------------------------------------


class TestCacheHashBackwardCompat:
    """Default args produce same hash as old 3-arg signature."""

    def test_defaults_match_old_behavior(self):
        h1 = compute_visibility_scope_hash("internal", 1, [])
        h2 = compute_visibility_scope_hash("internal", 1, [], user_role="user", curator_scope=None)
        assert h1 == h2


# ---------------------------------------------------------------------------
# Hash deterministic properties
# ---------------------------------------------------------------------------


class TestCacheHashDeterministic:
    """Hash must be deterministic and consistent."""

    def test_same_input_same_output(self):
        h1 = compute_visibility_scope_hash("internal", 1, [5, 10])
        h2 = compute_visibility_scope_hash("internal", 1, [5, 10])
        assert h1 == h2

    def test_hash_is_hex_string(self):
        h = compute_visibility_scope_hash("internal", 1, [])
        assert all(c in "0123456789abcdef" for c in h)

    def test_hash_length_consistent(self):
        h1 = compute_visibility_scope_hash("internal", 1, [])
        h2 = compute_visibility_scope_hash("client", 999, [1, 2, 3, 4, 5])
        assert len(h1) == len(h2)

    def test_hash_differs_for_different_identities(self):
        h1 = compute_visibility_scope_hash("internal", 1, [])
        h2 = compute_visibility_scope_hash("internal", 2, [])
        assert h1 != h2

    def test_hash_differs_for_different_kinds(self):
        h1 = compute_visibility_scope_hash("internal", 1, [])
        h2 = compute_visibility_scope_hash("client", 1, [])
        assert h1 != h2

    def test_hash_differs_for_different_groups(self):
        h1 = compute_visibility_scope_hash("internal", 1, [10])
        h2 = compute_visibility_scope_hash("internal", 1, [20])
        assert h1 != h2

    def test_hash_includes_groups_sorted(self):
        h1 = compute_visibility_scope_hash("internal", 1, [3, 1, 2])
        h2 = compute_visibility_scope_hash("internal", 1, [1, 2, 3])
        assert h1 == h2

    def test_hash_empty_groups_same_as_no_groups(self):
        h1 = compute_visibility_scope_hash("internal", 1, [])
        h2 = compute_visibility_scope_hash("internal", 1, [])
        assert h1 == h2

    def test_scope_ids_sorted(self):
        s1 = CuratorScope(managed_client_ids=(30, 10, 20))
        s2 = CuratorScope(managed_client_ids=(10, 20, 30))
        h1 = compute_visibility_scope_hash("internal", 1, [], user_role="curator", curator_scope=s1)
        h2 = compute_visibility_scope_hash("internal", 1, [], user_role="curator", curator_scope=s2)
        assert h1 == h2
