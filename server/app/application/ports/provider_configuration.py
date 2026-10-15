"""Provider validation and process configuration application status."""

from domain.entities.config_parameter import ConfigParameter
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ConfigurationStatus:
    saved_version: str | None
    applied_version: str | None
    ready: bool
    error: str | None


@dataclass(frozen=True)
class EffectiveParameter:
    key: str
    source: str
    configured_value: str | None
    applied_value: str | None
    editable: bool
    application_mode: str
    present: bool | None = None
    domain_key: str | None = None


class ProviderConfigurationPort(Protocol):
    def effective_parameters(self, rows: list[ConfigParameter]) -> list[EffectiveParameter]: ...

    async def validate_snapshot_change(
        self, rows: list[ConfigParameter], key: str, value: str, domain_key: str | None
    ) -> None: ...
    def validate_change(self, key: str, value: str) -> None: ...
    def validate_domain_change(self, key: str, value: str, domain_key: str) -> None: ...
    def status_for(self, rows: list[ConfigParameter]) -> ConfigurationStatus: ...
    def status(self) -> ConfigurationStatus: ...
