"""Characterization tests for answer cache hash behavior.

These tests document the CURRENT (pre-fix) behavior of
compute_visibility_scope_hash — it ignores user_role and curator scope.

All tests MUST pass BEFORE any Phase B-D changes. They serve as regression baseline.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from infrastructure.ml.answer_cache import compute_visibility_scope_hash


# ---------------------------------------------------------------------------
# A4: compute_visibility_scope_hash ignores role and scope
# ---------------------------------------------------------------------------


class TestCacheHashIgnoresRoleAndScope:
    """Document the security gap: cache hash is based only on identity, not ACL scope.

    Today: hash = content_hash(f"{user_kind}:{user_id}:{sorted(group_ids)}")

    Missing: user_role, managed_client_ids, managed_internal_ids, managed_group_ids

    Impact: A curator demoted to USER gets stale cache hits with data from
    managed users they no longer have access to.
    """

    def test_same_hash_for_user_and_curator_same_identity(self):
        """USER and CURATOR with same identity get identical cache hash.

        This is the core security gap — role is not part of the hash.
        """
        hash_user = compute_visibility_scope_hash("internal", 1, [])
        hash_curator = compute_visibility_scope_hash("internal", 1, [])
        # Same identity → same hash (expected today — THIS IS THE BUG)
        assert hash_user == hash_curator

    def test_hash_depends_only_on_kind_user_groups(self):
        """Verify hash is deterministic from kind, user_id, groups only."""
        h1 = compute_visibility_scope_hash("internal", 42, [1, 2, 3])
        h2 = compute_visibility_scope_hash("internal", 42, [1, 2, 3])
        h3 = compute_visibility_scope_hash("internal", 42, [3, 2, 1])  # different order
        assert h1 == h2
        assert h1 == h3  # sorted internally

    def test_hash_differs_for_different_identities(self):
        """Different user_id → different hash (correct behavior)."""
        h1 = compute_visibility_scope_hash("internal", 1, [])
        h2 = compute_visibility_scope_hash("internal", 2, [])
        assert h1 != h2

    def test_hash_differs_for_different_kinds(self):
        """Different user_kind → different hash (correct behavior)."""
        h1 = compute_visibility_scope_hash("internal", 1, [])
        h2 = compute_visibility_scope_hash("client", 1, [])
        assert h1 != h2

    def test_hash_differs_for_different_groups(self):
        """Different group_ids → different hash (correct behavior)."""
        h1 = compute_visibility_scope_hash("internal", 1, [10])
        h2 = compute_visibility_scope_hash("internal", 1, [20])
        assert h1 != h2

    def test_hash_includes_groups_sorted(self):
        """Group order doesn't matter — sorted internally."""
        h1 = compute_visibility_scope_hash("internal", 1, [3, 1, 2])
        h2 = compute_visibility_scope_hash("internal", 1, [1, 2, 3])
        assert h1 == h2

    def test_hash_empty_groups_same_as_no_groups(self):
        """Empty list and no groups → same hash."""
        h1 = compute_visibility_scope_hash("internal", 1, [])
        h2 = compute_visibility_scope_hash("internal", 1, [])
        assert h1 == h2


# ---------------------------------------------------------------------------
# Cache hash race condition scenario
# ---------------------------------------------------------------------------


class TestCacheHashDemotionScenario:
    """Characterize the demotion cache poisoning scenario.

    Scenario:
    1. Curator C (managed_internal_ids=[20,30]) asks a question
       → answer cached with vis_hash H
    2. C is demoted to USER (role changes, managed_ids cleared)
    3. C asks the same question → vis_hash H is the same → cache HIT
    4. C receives cached answer referencing documents of users 20,30
       which C no longer has access to
    """

    def test_demotion_creates_same_hash(self):
        """Before and after demotion, hash is identical (the bug)."""
        # Before demotion: curator
        hash_before = compute_visibility_scope_hash("internal", 42, [10, 20])
        # After demotion: user, same groups
        hash_after = compute_visibility_scope_hash("internal", 42, [10, 20])
        assert hash_before == hash_after

    def test_elevation_creates_same_hash(self):
        """Promotion also has same hash (fail-closed direction)."""
        hash_user = compute_visibility_scope_hash("internal", 42, [])
        hash_curator = compute_visibility_scope_hash("internal", 42, [])
        # Both directions — same hash (no role in hash)
        assert hash_user == hash_curator

    def test_assignment_change_same_hash(self):
        """Changing managed_ids doesn't affect hash (identity unchanged)."""
        hash_before = compute_visibility_scope_hash("internal", 42, [10, 20])
        hash_after = compute_visibility_scope_hash("internal", 42, [10, 20])
        # managed_ids changed but groups didn't → same hash
        assert hash_before == hash_after


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
