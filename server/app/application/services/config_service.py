"""Application Service: ConfigService — управление динамической конфигурацией.

Валидация min/max — бизнес-правило, поэтому здесь, а не в роутере.
publish() вызывается ПОСЛЕ выхода из `async with uow_factory.create()`,
то есть гарантированно после commit — подписчики не увидят "недокоммиченное" значение.
"""

from __future__ import annotations

import logging

from domain.events.config_events import ConfigParameterChanged
from domain.exceptions import EntityNotFound, ValidationError
from domain.repositories.config_parameter_repository import ConfigParameter
from domain.utils import parse_bool

from application.ports.config_broadcaster import ConfigChangeBroadcaster
from application.ports.event_bus import EventBus
from application.ports.unit_of_work_factory import UnitOfWorkFactory

log = logging.getLogger("default")


class ConfigService:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        event_bus: EventBus,
        broadcaster: ConfigChangeBroadcaster,
    ) -> None:
        self._uow_factory = uow_factory
        self._bus = event_bus
        self._broadcaster = broadcaster

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

            self._validate(param, raw_value)
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
            await self._broadcaster.broadcast_within_session(uow.session, event)
        # commit + NOTIFY happen atomically here

        self._bus.publish(event)

        return param

    @staticmethod
    def _validate(param: ConfigParameter, raw_value: str) -> None:
        if param.value_type == "bool":
            try:
                parse_bool(raw_value)
            except ValueError:
                raise ValidationError(f"Value for '{param.key}' must be boolean")
            return
        if param.value_type == "str":
            if param.allowed_values is not None and raw_value not in param.allowed_values:
                allowed = ", ".join(param.allowed_values)
                raise ValidationError(f"Value for '{param.key}' must be one of: {allowed}")
            return
        if param.value_type not in ("int", "float"):
            return
        try:
            val = int(raw_value) if param.value_type == "int" else float(raw_value)
        except ValueError as e:
            raise ValidationError(f"Invalid value for '{param.key}': {e}") from e
        if param.min_value is not None and val < param.min_value:
            raise ValidationError(f"Value for '{param.key}' must be >= {param.min_value}")
        if param.max_value is not None and val > param.max_value:
            raise ValidationError(f"Value for '{param.key}' must be <= {param.max_value}")
