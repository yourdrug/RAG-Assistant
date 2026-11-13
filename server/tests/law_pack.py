"""Belarusian law pack registry for tests of its document profiles and policies."""

from __future__ import annotations

from domain.ingestion.profiles.registry import DomainProfileRegistry
from infrastructure.domain_profile import register_pack_profiles
from infrastructure.domain_profile.settings_adapter import DomainSettingsAdapter
from packs.belarus_law.pack import BELARUS_LAW_PACK


def law_settings() -> DomainSettingsAdapter:
    """Domain settings seeded with the law pack profiles' declared defaults."""
    settings = DomainSettingsAdapter()
    settings.replace_snapshot(
        {
            (default.key, profile.key): default.value
            for profile in BELARUS_LAW_PACK.build_profiles(None)
            for default in profile.config_defaults()
        }
    )
    return settings


def law_registry(settings: DomainSettingsAdapter | None = None) -> DomainProfileRegistry:
    registry = DomainProfileRegistry()
    register_pack_profiles(registry, BELARUS_LAW_PACK, settings or law_settings())
    return registry
