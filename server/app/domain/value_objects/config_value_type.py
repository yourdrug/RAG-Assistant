"""Config parameter value types."""

from __future__ import annotations

from enum import StrEnum


class ConfigValueType(StrEnum):
    BOOL = "bool"
    INT = "int"
    FLOAT = "float"
    STR = "str"
