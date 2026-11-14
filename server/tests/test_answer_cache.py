"""Tests for answer cache hash behavior — Phase C.

compute_visibility_scope_hash now includes user_role and curator_scope,
closing the demotion cache poisoning vulnerability.
"""

from __future__ import annotations

import pytest

from config import settings
from domain.identity.user_context import CuratorScope
from infrastructure.ml.cache.answers import (
    CACHE_PREFIX,
    cache_key,
    compute_question_hash,
    compute_visibility_scope_hash,
)
from infrastructure.ml.cache.keys import generation_identity


# ---------------------------------------------------------------------------
# Cache prefix
# ---------------------------------------------------------------------------


class TestCachePrefix:
    def test_prefix_versions_document_citations_and_revision_fence(self) -> None:
        assert CACHE_PREFIX == "rag:cache:v6:"


# ---------------------------------------------------------------------------
# Phase C: hash includes role and curator_scope
# ---------------------------------------------------------------------------


class TestCacheHashIncludesRoleAndScope:
    """Cache hash now includes user_role and curator_scope."""

    def test_hash_includes_role(self) -> None:
        h_user = compute_visibility_scope_hash("internal", 1, [], user_role="user")
        h_curator = compute_visibility_scope_hash("internal", 1, [], user_role="curator")
        assert h_user != h_curator

    def test_hash_includes_curator_scope(self) -> None:
        scope = CuratorScope(managed_client_ids=(10, 20))
        h_with = compute_visibility_scope_hash("internal", 1, [], user_role="curator", curator_scope=scope)
        h_without = compute_visibility_scope_hash("internal", 1, [], user_role="curator")
        assert h_with != h_without

    def test_hash_no_scope_vs_none(self) -> None:
        h_none = compute_visibility_scope_hash("internal", 1, [], curator_scope=None)
        h_empty = compute_visibility_scope_hash("internal", 1, [], curator_scope=CuratorScope())
        assert h_none == h_empty

    def test_hash_empty_curator_scope(self) -> None:
        h1 = compute_visibility_scope_hash("internal", 1, [], user_role="curator")
        h2 = compute_visibility_scope_hash(
            "internal", 1, [], user_role="curator", curator_scope=CuratorScope()
        )
        assert h1 == h2

    def test_demotion_now_different_hash(self) -> None:
        scope = CuratorScope(managed_client_ids=(10, 20))
        h_before = compute_visibility_scope_hash(
            "internal", 42, [10, 20], user_role="curator", curator_scope=scope
        )
        h_after = compute_visibility_scope_hash("internal", 42, [10, 20], user_role="user")
        assert h_before != h_after

    def test_assignment_change_different_hash(self) -> None:
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

    def test_defaults_match_old_behavior(self) -> None:
        h1 = compute_visibility_scope_hash("internal", 1, [])
        h2 = compute_visibility_scope_hash("internal", 1, [], user_role="user", curator_scope=None)
        assert h1 == h2


# ---------------------------------------------------------------------------
# Hash deterministic properties
# ---------------------------------------------------------------------------


class TestCacheHashDeterministic:
    """Hash must be deterministic and consistent."""

    def test_same_input_same_output(self) -> None:
        h1 = compute_visibility_scope_hash("internal", 1, [5, 10])
        h2 = compute_visibility_scope_hash("internal", 1, [5, 10])
        assert h1 == h2

    def test_hash_is_hex_string(self) -> None:
        h = compute_visibility_scope_hash("internal", 1, [])
        assert all(c in "0123456789abcdef" for c in h)

    def test_hash_length_consistent(self) -> None:
        h1 = compute_visibility_scope_hash("internal", 1, [])
        h2 = compute_visibility_scope_hash("client", 999, [1, 2, 3, 4, 5])
        assert len(h1) == len(h2)

    def test_hash_differs_for_different_identities(self) -> None:
        h1 = compute_visibility_scope_hash("internal", 1, [])
        h2 = compute_visibility_scope_hash("internal", 2, [])
        assert h1 != h2

    def test_hash_differs_for_different_kinds(self) -> None:
        h1 = compute_visibility_scope_hash("internal", 1, [])
        h2 = compute_visibility_scope_hash("client", 1, [])
        assert h1 != h2

    def test_hash_differs_for_different_groups(self) -> None:
        h1 = compute_visibility_scope_hash("internal", 1, [10])
        h2 = compute_visibility_scope_hash("internal", 1, [20])
        assert h1 != h2

    def test_hash_includes_groups_sorted(self) -> None:
        h1 = compute_visibility_scope_hash("internal", 1, [3, 1, 2])
        h2 = compute_visibility_scope_hash("internal", 1, [1, 2, 3])
        assert h1 == h2

    def test_hash_empty_groups_same_as_no_groups(self) -> None:
        h1 = compute_visibility_scope_hash("internal", 1, [])
        h2 = compute_visibility_scope_hash("internal", 1, [])
        assert h1 == h2

    def test_scope_ids_sorted(self) -> None:
        s1 = CuratorScope(managed_client_ids=(30, 10, 20))
        s2 = CuratorScope(managed_client_ids=(10, 20, 30))
        h1 = compute_visibility_scope_hash("internal", 1, [], user_role="curator", curator_scope=s1)
        h2 = compute_visibility_scope_hash("internal", 1, [], user_role="curator", curator_scope=s2)
        assert h1 == h2


class TestAnswerCacheContracts:
    def test_same_scope_same_hash(self) -> None:
        h1 = compute_visibility_scope_hash("internal", 5, [1, 2])
        h2 = compute_visibility_scope_hash("internal", 5, [1, 2])
        assert h1 == h2

    def test_different_scope_different_hash(self) -> None:
        h1 = compute_visibility_scope_hash("internal", 5, [1])
        h2 = compute_visibility_scope_hash("client", 5, [1])
        assert h1 != h2

    def test_different_groups_different_hash(self) -> None:
        h1 = compute_visibility_scope_hash("internal", 5, [1, 2])
        h2 = compute_visibility_scope_hash("internal", 5, [1, 3])
        assert h1 != h2

    def test_different_users_same_scope_different_hash(self) -> None:
        # identical kind + groups must NOT share a cache scope —
        # answers/sources may reference the caller's own private documents.
        h1 = compute_visibility_scope_hash("internal", 1, [])
        h2 = compute_visibility_scope_hash("internal", 2, [])
        assert h1 != h2

    def test_question_hash_deterministic(self) -> None:
        h1 = compute_question_hash("Что такое ЭТТН?")
        h2 = compute_question_hash("Что такое ЭТТН?")
        assert h1 == h2

    def test_question_hash_case_insensitive(self) -> None:
        h1 = compute_question_hash("ЧТО ТАКОЕ ЭТТН?")
        h2 = compute_question_hash("что такое Эттн?")
        assert h1 == h2


class TestCacheScopeContracts:
    def test_visibility_scope_hash_deterministic(self) -> None:
        h1 = compute_visibility_scope_hash("internal_user", 42, [1, 2])
        h2 = compute_visibility_scope_hash("internal_user", 42, [1, 2])
        assert h1 == h2

    def test_different_users_different_hash(self) -> None:
        h1 = compute_visibility_scope_hash("internal_user", 1, [1])
        h2 = compute_visibility_scope_hash("internal_user", 2, [1])
        assert h1 != h2


def test_cache_key_changes_with_embedding_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "embedding_revision", "v1")
    first = cache_key("question", "same-acl")
    monkeypatch.setattr(settings, "embedding_revision", "v2")
    assert cache_key("question", "same-acl") != first


@pytest.mark.parametrize(
    "key,value",
    [
        ("llm_provider", "openrouter"),
        ("openai_compatible_model", "Qwen/Qwen2.5-1.5B-Instruct"),
        ("openai_compatible_base_url", "http://other-vllm:8000/v1"),
    ],
)
def test_question_hash_identity_changes_with_active_llm(
    monkeypatch: pytest.MonkeyPatch, key: str, value: str
) -> None:
    # generation_identity() is hashed into the question key: another model must miss the cache.
    monkeypatch.setattr(settings, "llm_provider", "openai_compatible")
    monkeypatch.setattr(settings, "openai_compatible_base_url", "http://vllm:8000/v1")
    monkeypatch.setattr(settings, "openai_compatible_model", "Qwen/Qwen2.5-3B-Instruct")
    first = generation_identity()
    monkeypatch.setattr(settings, key, value)
    assert generation_identity() != first
