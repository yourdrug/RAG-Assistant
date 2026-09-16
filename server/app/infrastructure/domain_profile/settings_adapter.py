"""DomainSettingsAdapter — reads domain-specific config from config_parameters table.

Implements DomainSettingsPort. Caches values in memory, invalidated by
ConfigParameterChanged events with matching domain_key.
"""

from __future__ import annotations

import logging

log = logging.getLogger("default")


class DomainSettingsAdapter:
    """Reads domain-specific config parameters from in-memory cache."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str], str] = {}

    def get(self, key: str, domain_key: str) -> str:
        """Get a config value for a specific domain.

        No fallback: every domain seeds and reads its own keys explicitly
        (see DomainProfile.config_defaults / seed_domain_config_defaults).
        Raises KeyError if the parameter is missing — a loud failure, never
        a silent fallback to a global value.
        """
        cache_key = (key, domain_key)
        if cache_key in self._cache:
            return self._cache[cache_key]
        raise KeyError(f"Config parameter '{key}' not found for domain '{domain_key}'")

    def set(self, key: str, domain_key: str | None, value: str) -> None:
        """Update cached value. Called by config event subscribers."""
        cache_key = (key, domain_key or "")
        self._cache[cache_key] = value
