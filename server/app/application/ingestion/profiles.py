"""Optional domain profile lookup shared by ingestion components."""

from __future__ import annotations

from typing import TYPE_CHECKING

from domain.knowledge.enums import DocDomain
from domain.shared.exceptions import ValidationError


if TYPE_CHECKING:
    from domain.ingestion.profiles.protocol import DocumentProfile
    from domain.ingestion.profiles.registry import DomainProfileRegistry


def get_domain_profile(registry: DomainProfileRegistry | None, domain: str) -> DocumentProfile | None:
    if registry is None:
        return None

    try:
        return registry.get(domain)
    except KeyError:
        return None


def known_domains(registry: DomainProfileRegistry | None) -> tuple[str, ...]:
    """Profile keys of the active domain pack; only ``general`` without a registry."""
    if registry is None:
        return (DocDomain.GENERAL.value,)

    return tuple(profile.key for profile in registry.all())


def ensure_known_domain(
    registry: DomainProfileRegistry | None, domain: str, *, field: str = "domain"
) -> None:
    """Reject a document profile key the active domain pack does not define."""
    known = known_domains(registry)

    if domain not in known:
        message = f"Unknown document domain {domain!r}; known: {', '.join(known)}"
        raise ValidationError(message, errors={field: message})
