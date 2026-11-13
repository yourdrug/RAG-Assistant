"""Domain profile infrastructure: registry population, settings adapter, answer policies.

Adding a domain = a new pack under ``packs/<name>/`` + one entry in the Composition
pack registry (``composition/domain_packs.py``); the core does not change.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from domain.ingestion.profiles.base import SettingsBackedProfile


if TYPE_CHECKING:
    from domain.ingestion.ports import DomainSettingsPort
    from domain.ingestion.profiles.registry import DomainProfileRegistry
    from infrastructure.domain_profile.packs import DomainPack


log = logging.getLogger("default")


def register_pack_profiles(
    registry: DomainProfileRegistry, pack: DomainPack, settings: DomainSettingsPort
) -> None:
    """Register the document profiles and document-type keywords of one domain pack.

    Validates that DomainSettingsPort is functional before registering
    settings-backed profiles so configuration errors are caught at startup,
    not on the first document processing request.
    """
    registry.doc_type_keywords = pack.doc_type_keywords

    for profile in pack.build_profiles(settings):
        if isinstance(profile, SettingsBackedProfile):
            try:
                profile.get_setting(f"{profile.key}_chunk_max_tokens")
            except (KeyError, RuntimeError):
                log.warning(
                    "DomainSettingsPort validation failed for %s -- config parameters may not be seeded yet",
                    type(profile).__name__,
                )

        registry.register(profile)
