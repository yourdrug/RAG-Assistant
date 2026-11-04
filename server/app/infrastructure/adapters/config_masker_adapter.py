"""Infrastructure adapter for ConfigMaskerPort — wraps config_subscribers."""

from __future__ import annotations

from application.runtime.setting_keys import SECRET_SETTING_KEYS
from infrastructure.ml.config.config_subscribers import mask_value


class ConfigMaskerAdapter:
    """Thin wrapper making config masking available as an injectable port."""

    @property
    def sensitive_keys(self) -> frozenset[str]:
        return SECRET_SETTING_KEYS

    def mask_value(self, value: str | None) -> str:
        return mask_value(value)
