"""Read answer and query policies from the registered document profiles."""

from __future__ import annotations

from typing import TYPE_CHECKING

from domain.ingestion.profiles.catalog import PROFILE_TYPES
from domain.rag.classification import classify_query_domain as classify_with_patterns
from domain.shared.profile_policy import AnswerPolicy


if TYPE_CHECKING:
    from domain.ingestion.profiles.protocol import DocumentProfile
    from domain.ingestion.profiles.registry import DomainProfileRegistry

DEFAULT_PROFILES: dict[str, DocumentProfile] = {
    profile.key: profile for cls in PROFILE_TYPES for profile in (cls(),)
}


def policy_for(domain_key: str, registry: DomainProfileRegistry | None = None) -> AnswerPolicy:
    profiles = (
        {profile.key: profile for profile in registry.all()} if registry is not None else DEFAULT_PROFILES
    )
    profile = profiles.get(domain_key)
    return profile.answer_policy if profile is not None else AnswerPolicy()


def classify_query_domain(question: str, registry: DomainProfileRegistry | None = None) -> str:
    profiles = registry.all() if registry is not None else list(DEFAULT_PROFILES.values())
    patterns = {
        profile.key: profile.query_patterns
        for profile in sorted(profiles, key=lambda profile: profile.query_priority, reverse=True)
    }
    return classify_with_patterns(question, patterns)
