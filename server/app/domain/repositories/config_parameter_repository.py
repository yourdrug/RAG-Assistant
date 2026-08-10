"""ConfigParameter repository interface -- CRUD for dynamic configuration parameters."""

from __future__ import annotations

from typing import Protocol


class ConfigParameter:
    __slots__ = (
        "key",
        "value",
        "value_type",
        "category",
        "description",
        "min_value",
        "max_value",
        "allowed_values",
    )

    def __init__(
        self,
        key: str,
        value: str,
        value_type: str,
        category: str,
        description: str | None = None,
        min_value: float | None = None,
        max_value: float | None = None,
        allowed_values: list[str] | None = None,
    ) -> None:
        self.key = key
        self.value = value
        self.value_type = value_type
        self.category = category
        self.description = description
        self.min_value = min_value
        self.max_value = max_value
        self.allowed_values = allowed_values


class ConfigParameterRepository(Protocol):
    async def get_all(self) -> list[ConfigParameter]: ...
    async def get_by_key(self, key: str) -> ConfigParameter | None: ...
    async def update_value(self, key: str, value: str) -> None: ...
