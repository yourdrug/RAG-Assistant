"""Config admin ports — abstract interfaces for system info providers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable


if TYPE_CHECKING:
    from application.admin.dto import ConfigurationStatus, EffectiveParameter
    from domain.entities.config_parameter import ConfigParameter


@runtime_checkable
class OllamaProbePort(Protocol):
    async def get_models(self) -> list[str]: ...


@runtime_checkable
class VectorDBInfoPort(Protocol):
    async def get_status(self) -> str: ...
    async def get_collections(self) -> list[dict[str, Any]]: ...


class ProviderConfigurationPort(Protocol):
    def effective_parameters(self, rows: list[ConfigParameter]) -> list[EffectiveParameter]: ...

    async def validate_snapshot_change(
        self, rows: list[ConfigParameter], key: str, value: str, domain_key: str | None
    ) -> None: ...
    def validate_change(self, key: str, value: str) -> None: ...
    def validate_domain_change(self, key: str, value: str, domain_key: str) -> None: ...
    def status_for(self, rows: list[ConfigParameter]) -> ConfigurationStatus: ...
    def status(self) -> ConfigurationStatus: ...


class ConfigMaskerPort(Protocol):
    """Masks sensitive configuration values."""

    @property
    def sensitive_keys(self) -> frozenset[str]: ...

    def mask_value(self, value: str | None) -> str: ...


class CorpusControlPort(Protocol):
    async def is_ready(self) -> bool: ...

    async def status(self) -> dict[str, Any]: ...

    async def switch(self, endpoint: str, collection: str, expected_generation: str) -> dict[str, Any]: ...
