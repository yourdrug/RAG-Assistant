"""ConfigParameterChanged subscribers -- hot-reload reactions for dynamic settings.

Each concern is a separate function; adding a new reaction means adding a new
subscribe() call without touching existing ones.

Dynamic (hot-reloadable) parameters are those listed in _DYNAMIC_FIELDS.
They can be changed via the API without a restart and are applied to the
in-memory settings immediately after a DB commit + NOTIFY.

Static parameters -- everything else in .env / Settings -- cannot be changed
without restarting the process (read once at startup).  When adding a new
parameter, decide: should it be hot-reloadable?  If yes, add it to
_DYNAMIC_FIELDS and the config_parameters DB table.  If no, leave it in
.env and require a restart.
"""

from __future__ import annotations

import json
import logging

from config import settings
from domain.runtime.config import ConfigParameterChanged
from domain.runtime.config import parse_bool

from infrastructure.ml.ingestion import get_paddle_ocr
from infrastructure.ml.guardrails.guardrails import invalidate_pii_detector

from domain.runtime.dynamic_params import DYNAMIC_FIELDS_MAP
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


_DYNAMIC_FIELDS: dict[str, tuple[str, type]] = DYNAMIC_FIELDS_MAP


def coerce_and_set(attr: str, expected_type: type | None, new_value: str) -> None:
    """Coerce *new_value* to *expected_type* and set it on the global settings."""
    if expected_type is bool:
        setattr(settings, attr, parse_bool(new_value))
    elif expected_type is int:
        setattr(settings, attr, int(new_value))
    elif expected_type is float:
        setattr(settings, attr, float(new_value))
    elif expected_type is list:
        setattr(settings, attr, json.loads(new_value))
    elif expected_type is str:
        raw = new_value
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, str):
                raw = parsed
        except (json.JSONDecodeError, TypeError):
            pass
        setattr(settings, attr, raw)
    else:
        setattr(settings, attr, new_value)


def apply_to_settings(event: ConfigParameterChanged) -> None:
    """Применить новое значение к in-memory settings."""
    if event.key.lower() in SENSITIVE_KEYS:
        return
    # Domain-specific params don't touch global settings — they live in DomainSettingsPort
    if getattr(event, "domain_key", None) is not None:
        return
    # Only declared hot-reloadable fields are applied. The startup resync
    # republishes EVERY row in config_parameters (including static ones like
    # embed_dim / file_backend); blindly setattr-ing those stored the DB's
    # string form over typed settings (e.g. embed_dim "1024" as str broke
    # every %-d log format that used it).
    if event.key not in _DYNAMIC_FIELDS:
        return
    attr, expected_type = _DYNAMIC_FIELDS[event.key]
    if not hasattr(settings, attr):
        return
    try:
        coerce_and_set(attr, expected_type, event.new_value)
        if event.key in SENSITIVE_KEYS:
            log.info("Sensitive config applied: %s", event.key)
        else:
            log.info("Config applied: %s = %s (was %s)", event.key, event.new_value, event.old_value)
    except (ValueError, TypeError) as e:
        if event.key in SENSITIVE_KEYS:
            log.warning("Failed to apply sensitive config %s: %s", event.key, e)
        else:
            log.warning("Failed to apply config %s=%r: %s", event.key, event.new_value, e)


# ---------------------------------------------------------------------------
# Cache invalidation subscribers
# ---------------------------------------------------------------------------


def invalidate_paddle_ocr_cache(event: ConfigParameterChanged) -> None:
    """Сбросить кэш PaddleOCR при смене языка (модель перезагрузится лениво)."""
    if event.key != "ocr_lang_paddle":
        return

    get_paddle_ocr.cache_clear()
    log.info("PaddleOCR cache invalidated (ocr_lang_paddle -> %s)", event.new_value)


def invalidate_storage_cache(event: ConfigParameterChanged, storage=None, ml_clients=None) -> None:
    """Сбросить кэш хранилища при изменении backend или S3-параметров."""
    # Storage is deployment-owned; legacy DB events cannot rotate credentials.
    return


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


def invalidate_pii_detector_cache(event: ConfigParameterChanged) -> None:
    """Сбросить кэш PII-детектора при изменении pii_redaction_enabled."""
    if event.key != "pii_redaction_enabled":
        return

    invalidate_pii_detector()
    log.info("PII detector cache invalidated (pii_redaction_enabled -> %s)", event.new_value)
