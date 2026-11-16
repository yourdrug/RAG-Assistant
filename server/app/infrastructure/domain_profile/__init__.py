"""Domain profile infrastructure: registry population, settings adapter, answer policies.

Adding a domain = a new pack under ``infrastructure/domain_packs/<name>/`` + one entry in the Composition
pack registry (``composition/domain_packs/__init__.py``); the core does not change.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from domain.ingestion.profiles.base import SettingsBackedProfile
from domain.ingestion.profiles.registry import DomainProfileRegistry


if TYPE_CHECKING:
    from collections.abc import Iterable

    from domain.ingestion.ports import DomainSettingsPort
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


def pack_registry(pack: DomainPack, settings: DomainSettingsPort) -> DomainProfileRegistry:
    """Document profiles of one pack: a space classifies and splits only with these."""
    registry = DomainProfileRegistry()
    register_pack_profiles(registry, pack, settings)
    return registry


def union_registry(registries: Iterable[DomainProfileRegistry]) -> DomainProfileRegistry:
    """Profiles of every loaded pack, for profile settings and admin tools, never for classification.

    A profile shared by several packs (``general``, ``contract``) has one key and one class.
    """
    union = DomainProfileRegistry()
    keywords: list[tuple[str, tuple[str, ...]]] = []

    for registry in registries:
        for profile in registry.all():
            union.register(profile)

        keywords.extend(item for item in registry.doc_type_keywords if item not in keywords)

    union.doc_type_keywords = tuple(keywords)
    return union
