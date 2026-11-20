"""Domain profile infrastructure: registry population, settings adapter, answer policies.

Adding a domain = a new pack under ``infrastructure/domain_packs/<name>/`` + one entry in the Composition
pack registry (``composition/domain_packs/__init__.py``); the core does not change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from domain.ingestion.profiles.registry import DomainProfileRegistry


if TYPE_CHECKING:
    from collections.abc import Iterable

    from domain.ingestion.ports import DomainSettingsPort
    from infrastructure.domain_profile.packs import DomainPack


def register_pack_profiles(
    registry: DomainProfileRegistry, pack: DomainPack, settings: DomainSettingsPort
) -> None:
    """Register the document profiles and document-type keywords of one domain pack.

    Profile settings are not read here: the settings snapshot is still empty at registration.
    Missing domain keys fail startup in ``bootstrap_configuration`` (``require_complete=True``).
    """
    registry.doc_type_keywords = pack.doc_type_keywords

    for profile in pack.build_profiles(settings):
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
