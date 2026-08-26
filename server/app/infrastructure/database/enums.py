"""Database infrastructure enums -- node role definitions."""

from enum import StrEnum


class DatabaseNodeRole(StrEnum):
    MASTER = "master"
    SLAVE = "slave"
