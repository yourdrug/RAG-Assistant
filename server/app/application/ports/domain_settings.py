"""DomainSettingsPort — per-domain config access protocol.

Domain-specific config parameters (max_unit_chars, classification_threshold, etc.)
are read through this port, never from global settings or hardcoded values.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class DomainSettingsPort(Protocol):
    def get(self, key: str, domain_key: str) -> str: ...
