"""Application service for managing dynamic configuration parameters.

Validates values via the ConfigParameter entity's validate() method.
``publish_event()`` is called within the UoW block; the infrastructure UoW
forwards the event to the broadcaster atomically with the transaction commit.
"""

from __future__ import annotations

import logging

from application.ports.event_bus import EventBus
from application.ports.provider_configuration import ProviderConfigurationPort, ConfigurationStatus
from application.ports.unit_of_work_factory import UnitOfWorkFactory
from domain.events.config_events import ConfigParameterChanged
from domain.exceptions import EntityNotFound, ValidationError
from domain.value_objects.dynamic_config import (
    DYNAMIC_FIELDS_MAP,
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

    async def configuration_status(self) -> ConfigurationStatus:
        if self._provider_configuration is None:
            return ConfigurationStatus(None, None, False, "Configuration not initialized")
        async with self._uow_factory.create(master=True) as uow:
            rows = await uow.config_parameters.get_all()
        return self._provider_configuration.status_for(rows)

    async def list_parameters(self) -> list[ConfigParameter]:
        async with self._uow_factory.create() as uow:
            rows = await uow.config_parameters.get_all()
        return [
            row
            for row in rows
            if row.key.lower() not in SECRET_SETTING_KEYS | STATIC_SETTING_KEYS
            and (row.domain_key is not None or row.key in DYNAMIC_FIELDS_MAP)
        ]

    async def update_parameter(
        self,
        key: str,
        raw_value: str,
        changed_by: int | None = None,
        domain_key: str | None = None,
    ) -> ConfigParameter:
        if key.lower() in SECRET_SETTING_KEYS | STATIC_SETTING_KEYS | METADATA_SETTING_KEYS:
            raise ValidationError(f"'{key}' is managed by deployment configuration")
        if domain_key is None and key not in DYNAMIC_FIELDS_MAP:
            raise ValidationError(f"'{key}' is not an active dynamic parameter")
        async with self._uow_factory.create(master=True) as uow:
            await uow.config_parameters.lock_configuration()
            param = await uow.config_parameters.get_by_key_and_domain(key, domain_key)
            if param is None:
                raise EntityNotFound("ConfigParameter", f"{key}[domain={domain_key or 'global'}]")

            normalized = param.normalize(raw_value)
            param.validate(normalized)
            if self._provider_configuration is not None:
                if domain_key is None:
                    self._provider_configuration.validate_change(key, normalized)
                else:
                    self._provider_configuration.validate_domain_change(key, normalized, domain_key)
            elif key in {"ml_provider", "llm_provider"}:
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
