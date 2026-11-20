"""Read answer and query policies from the registered document profiles."""

from __future__ import annotations

from typing import TYPE_CHECKING

from domain.ingestion.profiles.general import GeneralDomainProfile
from domain.rag.classification import classify_query_domain as classify_with_patterns, matching_query_domains
from domain.shared.profile_policy import AnswerPolicy


if TYPE_CHECKING:
    from domain.ingestion.profiles.protocol import DocumentProfile
    from domain.ingestion.profiles.registry import DomainProfileRegistry


# Without a registry only the neutral profile applies; pack policies come from the registry.
DEFAULT_PROFILES: dict[str, DocumentProfile] = {GeneralDomainProfile.key: GeneralDomainProfile()}


def policy_for(domain_key: str, registry: DomainProfileRegistry | None = None) -> AnswerPolicy:
    profiles = (
        {profile.key: profile for profile in registry.all()} if registry is not None else DEFAULT_PROFILES
    )
    profile = profiles.get(domain_key)
    return profile.answer_policy if profile is not None else AnswerPolicy()


def query_patterns(registry: DomainProfileRegistry | None) -> dict[str, tuple[str, ...]]:
    profiles = registry.all() if registry is not None else list(DEFAULT_PROFILES.values())
    return {
        profile.key: profile.query_patterns
        for profile in sorted(profiles, key=lambda profile: profile.query_priority, reverse=True)
    }


def classify_query_domain(question: str, registry: DomainProfileRegistry | None = None) -> str:
    return classify_with_patterns(question, query_patterns(registry))


def search_domains(question: str, domain: str, registry: DomainProfileRegistry | None = None) -> list[str]:
    """Domains of the first, domain-filtered search: the selected one plus other matched ones.

    A question naming acts of several kinds (a law and a decree) must not lose
    one of them to the domain filter; only profiles that also fall back to the
    full corpus are added, so the filter never widens into unrelated kinds.
    """
    others = [
        key
        for key in matching_query_domains(question, query_patterns(registry))
        if key != domain and policy_for(key, registry).fallback_to_full_corpus
    ]
    return [domain, *others]
