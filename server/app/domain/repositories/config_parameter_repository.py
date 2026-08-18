"""ConfigParameter repository interface -- CRUD for dynamic configuration parameters."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from domain.entities.config_parameter import ConfigParameter


@runtime_checkable
class ConfigParameterRepository(Protocol):
    async def get_all(self) -> list[ConfigParameter]: ...
    async def get_by_key(self, key: str) -> ConfigParameter | None: ...
    async def update_value(self, key: str, value: str) -> None: ...
