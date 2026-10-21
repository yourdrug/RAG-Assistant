"""Dynamic configuration values, validation and change events."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

from domain.shared.exceptions import ValidationError


def parse_bool(value: str) -> bool:
    """Parse a string to boolean, raising ValueError on failure."""
    if value.lower() in ("true", "1", "yes", "on"):
        return True

    if value.lower() in ("false", "0", "no", "off"):
        return False

    msg = f"Cannot parse '{value}' as boolean"
    raise ValueError(msg)


class ConfigValueType(StrEnum):
    BOOL = "bool"
    INT = "int"
    FLOAT = "float"
    STR = "str"
    LIST = "list"


@dataclass
class ConfigParameter:
    key: str
    value: str
    value_type: str
    category: str
    description: str | None = None
    min_value: float | None = None
    max_value: float | None = None
    allowed_values: list[str] | None = None
    domain_key: str | None = None  # NULL = global, set for domain-specific params

    def normalize(self, raw_value: str) -> str:
        """Strip whitespace and surrounding quotes from string values.

        For non-string types (int, float, bool) returns the value unchanged.
        """
        if self.value_type != ConfigValueType.STR:
            return raw_value

        return raw_value.strip().strip('"').strip("'")

    def validate(self, raw_value: str) -> None:
        """Validate a raw value against this parameter's constraints.

        Raises ValidationError if the value is invalid.
        """
        if self.value_type == ConfigValueType.BOOL:
            try:
                parse_bool(raw_value)
            except ValueError:
                msg = f"Value for '{self.key}' must be boolean"
                raise ValidationError(msg) from None
            return

        if self.value_type == ConfigValueType.STR:
            self.validate_string(raw_value)
            return

        if self.value_type not in (ConfigValueType.INT, ConfigValueType.FLOAT):
            return

        try:
            val = int(raw_value) if self.value_type == ConfigValueType.INT else float(raw_value)
        except ValueError as e:
            msg = f"Invalid value for '{self.key}': {e}"
            raise ValidationError(msg) from e

        if self.min_value is not None and val < self.min_value:
            msg = f"Value for '{self.key}' must be >= {self.min_value}"
            raise ValidationError(msg)

        if self.max_value is not None and val > self.max_value:
            msg = f"Value for '{self.key}' must be <= {self.max_value}"
            raise ValidationError(msg)

    def validate_string(self, raw_value: str) -> None:
        normalized = self.normalize(raw_value)

        if self.allowed_values is not None and normalized not in self.allowed_values:
            allowed = ", ".join(self.allowed_values)
            msg = f"Value for '{self.key}' must be one of: {allowed}"
            raise ValidationError(msg)


@dataclass(frozen=True)
class ConfigParameterChanged:
    """Published after a dynamic config parameter is committed to the database.

    Subscribers decide whether the changed key is relevant to them -- the
    event is broadcast generically and filtering happens inside each handler.
    """

    key: str
    old_value: str | None
    new_value: str
    value_type: str
    changed_by: int | None = None
    occurred_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    domain_key: str | None = None
