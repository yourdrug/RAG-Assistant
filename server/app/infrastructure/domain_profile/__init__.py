"""Domain profile infrastructure: registry + settings adapter + profile registration.

Adding a new domain = new profile file in domain/ingestion/profiles/ +
one register call in register_all_profiles() below.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from domain.ingestion.profiles.decree import DecreeDomainProfile
from domain.ingestion.profiles.general import GeneralDomainProfile
from domain.ingestion.profiles.legal import LegalDomainProfile

if TYPE_CHECKING:
    from domain.ingestion.ports import DomainSettingsPort
    from domain.ingestion.profiles.registry import DomainProfileRegistry

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
            profile.get_setting("max_unit_chars")
        except Exception:
            log.warning(
                "DomainSettingsPort validation failed for %s -- config parameters may not be seeded yet",
                cls.__name__,
            )
        registry.register(profile)
