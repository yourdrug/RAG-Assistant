"""Validate and apply a process-local DB snapshot without yielding mid-switch."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace

from application.ports.provider_configuration import ConfigurationStatus
from config import Settings, settings
from domain.entities.config_parameter import ConfigParameter
from domain.events.config_events import ConfigParameterChanged
from domain.exceptions import ValidationError
from domain.utils import parse_bool
from domain.value_objects.dynamic_config import DYNAMIC_PARAMS, SECRET_SETTING_KEYS, STATIC_SETTING_KEYS
from domain.value_objects.llm_provider import LLMProvider
from domain.value_objects.config_value_type import ConfigValueType
from infrastructure.events.in_process_event_bus import event_bus, InProcessEventBus
from infrastructure.domain_profile.settings_adapter import DomainSettingsAdapter
from domain.domain_profile.registry import DomainProfileRegistry


def decode_parameter(param: ConfigParameter, value: str):
    param.validate(value)
    if param.value_type == ConfigValueType.BOOL:
        return parse_bool(value)
    if param.value_type == ConfigValueType.INT:
        return int(value)
    if param.value_type == ConfigValueType.FLOAT:
        result = float(value)
        if not math.isfinite(result):
            raise ValidationError(f"Invalid configuration parameter '{param.key}'")
        return result
    if param.value_type == ConfigValueType.LIST:
        result = json.loads(value)
        if not isinstance(result, list) or any(not isinstance(item, str) for item in result):
            raise ValidationError(f"Invalid configuration parameter '{param.key}'")
        return result
    return param.normalize(value)


class SettingsProviderConfiguration:
    def __init__(
        self,
        model: Settings = settings,
        domain_settings: DomainSettingsAdapter | None = None,
        registry: DomainProfileRegistry | None = None,
        bus: InProcessEventBus = event_bus,
    ) -> None:
        self.settings = model
        self.domain_settings = domain_settings
        self.bus = bus
        self.definitions = {}
        for declaration in DYNAMIC_PARAMS:
            self.definitions[(declaration.key, declaration.domain_key)] = ConfigParameter(
                declaration.key,
                "",
                declaration.type.__name__,
                declaration.category,
                min_value=declaration.min_val,
                max_value=declaration.max_val,
                allowed_values=declaration.allowed,
                domain_key=declaration.domain_key,
            )
        if registry is not None:
            for profile in registry.all():
                for default in profile.config_defaults():
                    self.definitions[(default.key, profile.key)] = ConfigParameter(
                        default.key,
                        default.value,
                        default.value_type,
                        f"domain:{profile.key}",
                        min_value=default.min_value,
                        max_value=default.max_value,
                        domain_key=profile.key,
                    )
        self._status = ConfigurationStatus(None, None, False, None)
        self.applying = False
        self.require_complete = False

    def status(self) -> ConfigurationStatus:
        return self._status

    def snapshot_version(self, rows: list[ConfigParameter]) -> str:
        active = [
            row
            for row in rows
            if (row.key, row.domain_key) in self.definitions
            and row.key.lower() not in SECRET_SETTING_KEYS | STATIC_SETTING_KEYS
        ]
        return hashlib.sha256(
            json.dumps(
                sorted((row.key, row.domain_key or "", row.value_type, row.value) for row in active),
                ensure_ascii=False,
            ).encode()
        ).hexdigest()

    def status_for(self, rows: list[ConfigParameter]) -> ConfigurationStatus:
        version = self.snapshot_version(rows)
        return replace(
            self._status,
            saved_version=version,
            ready=self._status.ready and version == self._status.applied_version,
        )

    def validate_provider(self, candidate: Settings) -> None:
        errors = candidate.check_ml_provider()
        if candidate.llm_provider == LLMProvider.OPENROUTER.value and not candidate.openrouter_api_key:
            errors.append("OPENROUTER_API_KEY required when LLM_PROVIDER=openrouter")
        if candidate.llm_provider not in {provider.value for provider in LLMProvider}:
            errors.append("Unknown LLM provider")
        if errors:
            raise ValidationError("; ".join(errors))

    def validate_domain_change(self, key: str, value: str, domain_key: str) -> None:
        definition = self.definitions.get((key, domain_key))
        if definition is None:
            raise ValidationError(f"'{key}' is not an active parameter for '{domain_key}'")
        try:
            decode_parameter(definition, value)
        except (ValueError, TypeError):
            raise ValidationError(f"Invalid configuration parameter '{key}'") from None

    def validate_change(self, key: str, value: str) -> None:
        definition = self.definitions.get((key, None))
        if definition is None:
            raise ValidationError(f"'{key}' is not an active global parameter")
        try:
            decoded = decode_parameter(definition, value)
        except (ValueError, TypeError) as exc:
            raise ValidationError(f"Invalid configuration parameter '{key}'") from exc
        candidate = self.settings.model_copy(update={key: decoded})
        self.validate_provider(candidate)

    def prepare_snapshot(self, active: list[ConfigParameter], require_complete: bool):
        pairs = [(row.key, row.domain_key) for row in active]
        if len(pairs) != len(set(pairs)) or (require_complete and set(pairs) != set(self.definitions)):
            raise ValidationError("Configuration snapshot is incomplete or contains duplicate keys")
        updates, domain_updates, events = {}, {}, []
        for row in active:
            definition = self.definitions[(row.key, row.domain_key)]
            if row.value_type != definition.value_type:
                raise ValidationError(f"Invalid configuration type for '{row.key}'")
            try:
                value = decode_parameter(definition, row.value)
            except (ValueError, TypeError) as exc:
                raise ValidationError(f"Invalid configuration parameter '{row.key}'") from exc
            if row.domain_key is None:
                old = getattr(self.settings, row.key)
                updates[row.key] = value
            else:
                if self.domain_settings is None:
                    raise ValidationError("Domain settings are not initialized")
                try:
                    old = self.domain_settings.get(row.key, row.domain_key)
                except KeyError:
                    old = None
                domain_updates[(row.key, row.domain_key)] = definition.normalize(row.value)
            if old != (value if row.domain_key is None else domain_updates[(row.key, row.domain_key)]):
                events.append(
                    ConfigParameterChanged(
                        key=row.key,
                        old_value=str(old) if old is not None else None,
                        new_value=definition.normalize(row.value),
                        value_type=row.value_type,
                        domain_key=row.domain_key,
                    )
                )
        self.validate_provider(self.settings.model_copy(update=updates))
        return updates, domain_updates, events

    def apply(self, rows: list[ConfigParameter], *, require_complete: bool = False) -> None:
        require_complete = require_complete or self.require_complete
        active = [
            row
            for row in rows
            if (row.key, row.domain_key) in self.definitions
            and row.key.lower() not in SECRET_SETTING_KEYS | STATIC_SETTING_KEYS
        ]
        version = self.snapshot_version(active)
        self._status = replace(self._status, saved_version=version, ready=False, error=None)
        previous_global = {}
        previous_domain = self.domain_settings.snapshot() if self.domain_settings is not None else {}
        try:
            updates, domain_updates, events = self.prepare_snapshot(active, require_complete)
            previous_global = {key: getattr(self.settings, key) for key in updates}
            # No awaits: all global/domain values become visible before any reaction runs.
            self.settings.__dict__.update(updates)
            if domain_updates and self.domain_settings is not None:
                self.domain_settings.replace_snapshot({**previous_domain, **domain_updates})
            self.applying = True
            for event in events:
                self.bus.publish_strict(event)
            self._status = ConfigurationStatus(version, version, True, None)
            self.require_complete = require_complete
        except Exception:
            self.settings.__dict__.update(previous_global)
            if self.domain_settings is not None:
                self.domain_settings.replace_snapshot(previous_domain)
            # Do not expose exception text: provider/subscriber errors can contain credentials.
            self._status = replace(
                self._status, ready=False, error="Configuration snapshot application failed"
            )
            raise
        finally:
            self.applying = False

    def failed(self) -> None:
        self._status = replace(
            self._status, ready=False, error="Configuration database synchronization failed"
        )
