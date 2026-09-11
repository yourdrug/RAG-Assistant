"""Domain profile infrastructure: registry + settings adapter + profile registration.

Adding a new domain = new profile file in domain/domain_profile/profiles/ +
one register call in register_all_profiles() below.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from domain.domain_profile.profiles.decree import DecreeDomainProfile
from domain.domain_profile.profiles.general import GeneralDomainProfile
from domain.domain_profile.profiles.legal import LegalDomainProfile

if TYPE_CHECKING:
    from application.ports.domain_settings import DomainSettingsPort
    from domain.domain_profile.registry import DomainProfileRegistry

log = logging.getLogger("default")


def register_all_profiles(registry: DomainProfileRegistry, settings: DomainSettingsPort) -> None:
    """Register all known domain profiles into the registry.

    Validates that DomainSettingsPort is functional before registering
    settings-backed profiles so configuration errors are caught at startup,
    not on the first document processing request.
    """
    registry.register(GeneralDomainProfile())

    for cls in (LegalDomainProfile, DecreeDomainProfile):
        profile = cls(settings=settings)
        try:
            profile._get("max_unit_chars")
        except Exception:
            log.warning(
                "DomainSettingsPort validation failed for %s -- " "config parameters may not be seeded yet",
                cls.__name__,
            )
        registry.register(profile)
