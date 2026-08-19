"""Application service for managing dynamic configuration parameters.

Validates values via the ConfigParameter entity's validate() method.
``publish_event()`` is called within the UoW block; the infrastructure UoW
forwards the event to the broadcaster atomically with the transaction commit.
"""

from __future__ import annotations

import logging

from domain.events.config_events import ConfigParameterChanged
from domain.exceptions import EntityNotFound
from domain.repositories.config_parameter_repository import ConfigParameter

from application.ports.event_bus import EventBus
from application.ports.unit_of_work_factory import UnitOfWorkFactory

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
        self, key: str, raw_value: str, changed_by: int | None = None
    ) -> ConfigParameter:
        async with self._uow_factory.create(master=True) as uow:
            param = await uow.config_parameters.get_by_key(key)
            if param is None:
                raise EntityNotFound("ConfigParameter", key)

            param.validate(raw_value)
            old_value = param.value
            await uow.config_parameters.update_value(key, raw_value)
            param.value = raw_value

            event = ConfigParameterChanged(
                key=key,
                old_value=old_value,
                new_value=raw_value,
                value_type=param.value_type,
                changed_by=changed_by,
            )
            await uow.publish_event(event)
        # commit + NOTIFY happen atomically here

        self._bus.publish(event)

        return param
