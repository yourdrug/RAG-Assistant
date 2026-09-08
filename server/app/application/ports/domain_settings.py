"""DomainSettingsPort -- re-export from domain layer.

The canonical definition lives in ``domain.domain_profile.settings_port``.
This module re-exports it for backward compatibility so that existing
imports (``from application.ports.domain_settings import DomainSettingsPort``)
continue to work.
"""

from __future__ import annotations

from domain.domain_profile.settings_port import DomainSettingsPort

__all__ = ["DomainSettingsPort"]
