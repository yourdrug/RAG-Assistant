"""Application service for managing dynamic configuration parameters.

Validates values via the ConfigParameter entity's validate() method.
``publish_event()`` is called within the UoW block; the infrastructure UoW
forwards the event to the broadcaster atomically with the transaction commit.
"""

from __future__ import annotations

import logging
from dataclasses import replace

from application.ports.event_bus import EventBus
from application.ports.provider_configuration import (
    ProviderConfigurationPort,
    ConfigurationStatus,
    EffectiveParameter,
)
from application.ports.unit_of_work_factory import UnitOfWorkFactory
from domain.events.config_events import ConfigParameterChanged
from domain.exceptions import EntityNotFound, ValidationError
from domain.value_objects.dynamic_config import (
    DYNAMIC_FIELDS_MAP,
    DYNAMIC_PARAMS,
    SECRET_SETTING_KEYS,
    STATIC_SETTING_KEYS,
    METADATA_SETTING_KEYS,
)
from domain.repositories.config_parameter_repository import ConfigParameter

log = logging.getLogger("default")


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
            raise ValidationError("Configuration not initialized")
        async with self._uow_factory.create(master=True) as uow:
            rows = await uow.config_parameters.get_all()
        return self._provider_configuration.effective_parameters(rows)

    async def configuration_status(self) -> ConfigurationStatus:
        if self._provider_configuration is None:
            return ConfigurationStatus(None, None, False, "Configuration not initialized")
        async with self._uow_factory.create(master=True) as uow:
            rows = await uow.config_parameters.get_all()
        return self._provider_configuration.status_for(rows)

    async def list_parameters(self) -> list[ConfigParameter]:
        async with self._uow_factory.create() as uow:
            rows = await uow.config_parameters.get_all()
        declarations = {(param.key, param.domain_key): param for param in DYNAMIC_PARAMS}
        result = []
        for row in rows:
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
        validate_corpus_update_key(key)
        if key.lower() in SECRET_SETTING_KEYS | STATIC_SETTING_KEYS | METADATA_SETTING_KEYS:
            raise ValidationError(f"'{key}' is managed by deployment configuration")
        if domain_key is None and key not in DYNAMIC_FIELDS_MAP:
            raise ValidationError(f"'{key}' is not an active dynamic parameter")
        validated_version = None
        if self._provider_configuration is not None:
            # Network preparation must finish before opening the write transaction.
            async with self._uow_factory.create(master=True) as uow:
                rows = await uow.config_parameters.get_all()
            if not any((row.key, row.domain_key) == (key, domain_key) for row in rows):
                raise EntityNotFound("ConfigParameter", f"{key}[domain={domain_key or 'global'}]")
            validated_version = self._provider_configuration.status_for(rows).saved_version
            await self._provider_configuration.validate_snapshot_change(rows, key, raw_value, domain_key)
        async with self._uow_factory.create(master=True) as uow:
            await uow.config_parameters.lock_configuration()
            if self._provider_configuration is not None:
                latest = await uow.config_parameters.get_all()
                if self._provider_configuration.status_for(latest).saved_version != validated_version:
                    raise ValidationError("Configuration changed during preparation; reload and retry")
            param = await uow.config_parameters.get_by_key_and_domain(key, domain_key)
            if param is None:
                raise EntityNotFound("ConfigParameter", f"{key}[domain={domain_key or 'global'}]")

            normalized = param.normalize(raw_value)
            param.validate(normalized)
            declaration = next(
                (p for p in DYNAMIC_PARAMS if (p.key, p.domain_key) == (key, domain_key)), None
            )
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
                raise ValidationError("Provider configuration validation is required")
            old_value = param.value

            await uow.config_parameters.update_value(key, normalized, domain_key=domain_key)
            param.value = normalized

            event = ConfigParameterChanged(
                key=key,
                old_value=old_value,
                new_value=normalized,
                value_type=param.value_type,
                changed_by=changed_by,
                domain_key=domain_key if domain_key else param.domain_key,
            )
            await uow.publish_event(event)

        self._bus.publish(event)
        return param


def validate_corpus_update_key(key: str) -> None:
    if key in {"qdrant_url", "collection_name"}:
        raise ValidationError("Use the coordinated corpus switch operation")
