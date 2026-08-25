"""Database infrastructure enums -- job status and role definitions."""

from enum import StrEnum


class DatabaseNodeRole(StrEnum):
    MASTER = "master"
    SLAVE = "slave"
