"""Application service for managing dynamic configuration parameters.

Validates values via the ConfigParameter entity's validate() method.
``publish_event()`` is called within the UoW block; the infrastructure UoW
forwards the event to the broadcaster atomically with the transaction commit.
"""

from __future__ import annotations

import logging

from application.ports.event_bus import EventBus
from application.ports.unit_of_work_factory import UnitOfWorkFactory
from domain.events.config_events import ConfigParameterChanged
from domain.exceptions import EntityNotFound
from domain.repositories.config_parameter_repository import ConfigParameter

log = logging.getLogger("default")


class ConfigService:
    def __init__(
            self,
            uow_factory: UnitOfWorkFactory,
            event_bus: EventBus,
    ) -> None:
        self._uow_factory = uow_factory
        self._bus = event_bus

    async def list_parameters(self) -> list[ConfigParameter]:
        async with self._uow_factory.create() as uow:
            return await uow.config_parameters.get_all()

    async def update_parameter(
            self,
            key: str,
            raw_value: str,
            changed_by: int | None = None,
            domain_key: str | None = None,
    ) -> ConfigParameter:
        async with self._uow_factory.create(master=True) as uow:
            param = await uow.config_parameters.get_by_key_and_domain(key, domain_key)
            if param is None:
                raise EntityNotFound("ConfigParameter", f"{key}[domain={domain_key or 'global'}]")

            normalized = param.normalize(raw_value)
            param.validate(normalized)
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
