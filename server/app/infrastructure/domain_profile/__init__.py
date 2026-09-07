"""Domain profile infrastructure: registry + settings adapter + profile registration.

Adding a new domain = new profile file in domain/domain_profile/profiles/ +
one register call in register_all_profiles() below.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from application.ports.domain_settings import DomainSettingsPort
    from infrastructure.domain_profile.registry import DomainProfileRegistry


def register_all_profiles(registry: DomainProfileRegistry, settings: DomainSettingsPort) -> None:
    """Register all known domain profiles into the registry."""
    from domain.domain_profile.profiles.decree import DecreeDomainProfile
    from domain.domain_profile.profiles.general import GeneralDomainProfile
    from domain.domain_profile.profiles.legal import LegalDomainProfile

    registry.register(GeneralDomainProfile())
    registry.register(LegalDomainProfile(settings=settings))
    registry.register(DecreeDomainProfile(settings=settings))
