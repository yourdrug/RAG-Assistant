"""Optional domain profile lookup shared by ingestion components."""

from __future__ import annotations

from typing import TYPE_CHECKING


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
