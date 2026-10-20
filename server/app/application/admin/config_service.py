"""Application service for managing dynamic configuration parameters.

Validates values via the ConfigParameter entity's validate() method.
``publish_event()`` is called within the UoW block; the infrastructure UoW
forwards the event to the broadcaster atomically with the transaction commit.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from application.admin.dto import ConfigurationStatus
from domain.events.config_events import ConfigParameterChanged
from domain.shared.exceptions import EntityNotFound, ValidationError
from domain.value_objects.dynamic_config import (
    DYNAMIC_FIELDS_MAP,
    DYNAMIC_PARAMS,
    METADATA_SETTING_KEYS,
    SECRET_SETTING_KEYS,
    STATIC_SETTING_KEYS,
)


if TYPE_CHECKING:
    from application.admin.dto import EffectiveParameter
    from application.admin.ports import ProviderConfigurationPort
    from application.ports.event_bus import EventBus
    from application.uow import UnitOfWorkFactory
    from domain.entities.config_parameter import ConfigParameter


class ConfigService:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        event_bus: EventBus,
        provider_configuration: ProviderConfigurationPort | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._bus = event_bus
        self._provider_configuration = provider_configuration

    async def effective_parameters(self) -> list[EffectiveParameter]:
        if self._provider_configuration is None:
            message = "Configuration not initialized"
            raise ValidationError(message)

        async with self._uow_factory.create(master=True) as uow:
            rows = await uow.config_parameters.get_all()
        return self._provider_configuration.effective_parameters(rows)

    async def configuration_status(self) -> ConfigurationStatus:
        if self._provider_configuration is None:
            return ConfigurationStatus(
                saved_version=None, applied_version=None, ready=False, error="Configuration not initialized"
            )

        async with self._uow_factory.create(master=True) as uow:
            rows = await uow.config_parameters.get_all()
        return self._provider_configuration.status_for(rows)

    async def list_parameters(self) -> list[ConfigParameter]:
        async with self._uow_factory.create() as uow:
            rows = await uow.config_parameters.get_all()
        declarations = {(param.key, param.domain_key): param for param in DYNAMIC_PARAMS}
        result = []

        for stored_row in rows:
            row = stored_row

            if row.key.lower() in SECRET_SETTING_KEYS | STATIC_SETTING_KEYS:
                continue

            if row.domain_key is None and row.key not in DYNAMIC_FIELDS_MAP:
                continue

            declaration = declarations.get((row.key, row.domain_key))

            if declaration is not None:
                row = replace(
                    row,
                    category=declaration.category,
                    description=declaration.description,
                    min_value=declaration.min_val,
                    max_value=declaration.max_val,
                    allowed_values=declaration.allowed,
                )

            result.append(row)

        return result

    async def update_parameter(
        self,
        key: str,
        raw_value: str,
        changed_by: int | None = None,
        domain_key: str | None = None,
    ) -> ConfigParameter:
        validate_update_key(key, domain_key)
        validated_version = await self.prepare_update(key, raw_value, domain_key)
        async with self._uow_factory.create(master=True) as uow:
            await uow.config_parameters.lock_configuration()

            if self._provider_configuration is not None:
                latest = await uow.config_parameters.get_all()

                if self._provider_configuration.status_for(latest).saved_version != validated_version:
                    message = "Configuration changed during preparation; reload and retry"
                    raise ValidationError(message)

            param = await uow.config_parameters.get_by_key_and_domain(key, domain_key)

            if param is None:
                entity_name = "ConfigParameter"
                raise EntityNotFound(entity_name, f"{key}[domain={domain_key or 'global'}]")

            normalized = self.normalize_parameter(param, raw_value, key, domain_key)
            old_value = param.value

            await uow.config_parameters.update_value(key, normalized, domain_key=domain_key)
            param.value = normalized

            event = ConfigParameterChanged(
                key=key,
                old_value=old_value,
                new_value=normalized,
                value_type=param.value_type,
                changed_by=changed_by,
                domain_key=domain_key or param.domain_key,
            )
            await uow.publish_event(event)

        self._bus.publish(event)
        return param

    async def prepare_update(self, key: str, raw_value: str, domain_key: str | None) -> str | None:
        validated_version = None

        if self._provider_configuration is not None:
            # Network preparation must finish before opening the write transaction.
            async with self._uow_factory.create(master=True) as uow:
                rows = await uow.config_parameters.get_all()

            if not any((row.key, row.domain_key) == (key, domain_key) for row in rows):
                entity_name = "ConfigParameter"
                raise EntityNotFound(entity_name, f"{key}[domain={domain_key or 'global'}]")

            validated_version = self._provider_configuration.status_for(rows).saved_version
            await self._provider_configuration.validate_snapshot_change(rows, key, raw_value, domain_key)

        return validated_version

    def normalize_parameter(
        self, param: ConfigParameter, raw_value: str, key: str, domain_key: str | None
    ) -> str:
        normalized = param.normalize(raw_value)
        param.validate(normalized)
        declaration = next((p for p in DYNAMIC_PARAMS if (p.key, p.domain_key) == (key, domain_key)), None)

        if declaration is not None:
            replace(
                param,
                min_value=declaration.min_val,
                max_value=declaration.max_val,
                allowed_values=declaration.allowed,
            ).validate(normalized)

        if self._provider_configuration is None and (
            key in {"ml_provider", "llm_provider", "deepinfra_embed_model", "deepinfra_rerank_model"}
            or key.endswith("_base_url")
            or key in {"tei_embed_url", "tei_rerank_url"}
            or "_http_" in key
        ):
            message = "Provider configuration validation is required"
            raise ValidationError(message)

        return normalized


def validate_corpus_update_key(key: str) -> None:
    if key in {"qdrant_url", "collection_name"}:
        message = "Use the coordinated corpus switch operation"
        raise ValidationError(message)


def validate_update_key(key: str, domain_key: str | None) -> None:
    validate_corpus_update_key(key)

    if key.lower() in SECRET_SETTING_KEYS | STATIC_SETTING_KEYS | METADATA_SETTING_KEYS:
        message = f"'{key}' is managed by deployment configuration"
        raise ValidationError(message)

    if domain_key is None and key not in DYNAMIC_FIELDS_MAP:
        message = f"'{key}' is not an active dynamic parameter"
        raise ValidationError(message)
