"""Application service for managing dynamic configuration parameters.

Validates values via the ConfigParameter entity's validate() method.
``publish_event()`` is called within the UoW block; the infrastructure UoW
forwards the event to the broadcaster atomically with the transaction commit.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from application.admin.dto import ConfigurationStatus
from application.runtime.config import ConfigParameterChanged
from application.runtime.dynamic_params import (
    DYNAMIC_FIELDS_MAP,
    DYNAMIC_PARAMS,
)
from application.runtime.setting_keys import (
    METADATA_SETTING_KEYS,
    SECRET_SETTING_KEYS,
    STATIC_SETTING_KEYS,
)
from domain.shared.exceptions import EntityNotFoundError, ValidationError


if TYPE_CHECKING:
    from application.admin.dto import EffectiveParameter
    from application.admin.ports import ProviderConfigurationPort
    from application.ports.event_bus import EventBus
    from application.runtime.config import ConfigParameter
    from application.uow import UnitOfWork, UnitOfWorkFactory


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
        (param,) = await self.update_parameters(
            {key: raw_value}, changed_by=changed_by, domain_key=domain_key
        )
        return param

    async def update_parameters(
        self,
        values: dict[str, str],
        changed_by: int | None = None,
        domain_key: str | None = None,
    ) -> list[ConfigParameter]:
        """Validate and store several parameters as one change: all of them or none.

        Linked limits (e.g. top_k <= fetch_k) are checked against each other's new values, so a
        valid pair is never rejected half-way because of the order in which it is written.
        """
        if not values:
            message = "No configuration parameters to update"
            raise ValidationError(message)

        for key in values:
            validate_update_key(key, domain_key)

        validated_version = await self.prepare_updates(values, domain_key)
        events = []
        params = []
        async with self._uow_factory.create(master=True) as uow:
            await uow.config_parameters.lock_configuration()
            await self.ensure_unchanged(uow, validated_version)
            # Every value is normalized before the first write; the transaction stores all or none.
            normalized = await self.normalized_values(uow, values, domain_key)

            for key, (param, value) in normalized.items():
                old_value = param.value
                await uow.config_parameters.update_value(key, value, domain_key=domain_key)
                param.value = value
                event = ConfigParameterChanged(
                    key=key,
                    old_value=old_value,
                    new_value=value,
                    value_type=param.value_type,
                    changed_by=changed_by,
                    domain_key=domain_key or param.domain_key,
                )
                await uow.publish_event(event)
                events.append(event)
                params.append(param)

        for event in events:
            self._bus.publish(event)

        return params

    async def ensure_unchanged(self, uow: UnitOfWork, validated_version: str | None) -> None:
        if self._provider_configuration is None:
            return

        latest = await uow.config_parameters.get_all()

        if self._provider_configuration.status_for(latest).saved_version != validated_version:
            message = "Configuration changed during preparation; reload and retry"
            raise ValidationError(message)

    async def normalized_values(
        self, uow: UnitOfWork, values: dict[str, str], domain_key: str | None
    ) -> dict[str, tuple[ConfigParameter, str]]:
        normalized: dict[str, tuple[ConfigParameter, str]] = {}

        for key, raw_value in values.items():
            param = await uow.config_parameters.get_by_key_and_domain(key, domain_key)

            if param is None:
                entity_name = "ConfigParameter"
                raise EntityNotFoundError(entity_name, f"{key}[domain={domain_key or 'global'}]")

            normalized[key] = (param, self.normalize_parameter(param, raw_value, key, domain_key))

        return normalized

    async def prepare_updates(self, values: dict[str, str], domain_key: str | None) -> str | None:
        validated_version = None

        if self._provider_configuration is not None:
            # Network preparation must finish before opening the write transaction.
            async with self._uow_factory.create(master=True) as uow:
                rows = await uow.config_parameters.get_all()

            present = {(row.key, row.domain_key) for row in rows}

            for key in values:
                if (key, domain_key) not in present:
                    entity_name = "ConfigParameter"
                    raise EntityNotFoundError(entity_name, f"{key}[domain={domain_key or 'global'}]")

            validated_version = self._provider_configuration.status_for(rows).saved_version
            await self._provider_configuration.validate_snapshot_changes(
                rows, {(key, domain_key): value for key, value in values.items()}
            )

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
            key in {"rerank_model", "rerank_api_format"} or key.endswith("_base_url") or "_http_" in key
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
