"""Audit and cache reactions after validated configuration snapshot publication."""

from __future__ import annotations

import logging

from domain.runtime.config import ConfigParameterChanged


from domain.runtime.setting_keys import STATIC_SETTING_KEYS
from domain.runtime.setting_keys import SECRET_SETTING_KEYS as SENSITIVE_KEYS

log = logging.getLogger("default")


def mask_value(value: str | None) -> str:
    """Return masked representation for sensitive values."""
    if value is None:
        return "None"

    if len(value) <= 4:
        return "****"

    return value[:2] + "*" * (len(value) - 4) + value[-2:]


def audit_log_config_change(event: ConfigParameterChanged) -> None:
    """Независимый аудит-лог — не зависит от settings/кэшей."""
    if event.key.lower() in SENSITIVE_KEYS | STATIC_SETTING_KEYS:
        log.info("AUDIT ignored deployment-owned config event key=%s", event.key)
        return

    log.info(
        "AUDIT config_change key=%s old=%r new=%r by_user=%s at=%s",
        event.key,
        event.old_value,
        event.new_value,
        event.changed_by,
        event.occurred_at,
    )
