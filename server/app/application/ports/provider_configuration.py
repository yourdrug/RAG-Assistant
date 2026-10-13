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


class ProviderConfigurationPort(Protocol):
    def validate_change(self, key: str, value: str) -> None: ...
    def validate_domain_change(self, key: str, value: str, domain_key: str) -> None: ...
    def status_for(self, rows: list[ConfigParameter]) -> ConfigurationStatus: ...
    def status(self) -> ConfigurationStatus: ...
