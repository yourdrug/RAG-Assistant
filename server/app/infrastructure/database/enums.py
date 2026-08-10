"""Database infrastructure enums -- environment, job status, and role definitions."""

from enum import Enum, StrEnum


class Environment(Enum):
    PRODUCTION = "PROD"
    TESTING = "TEST"
    DEVELOPMENT = "DEV"


class DatabaseNodeRole(StrEnum):
    MASTER = "master"
    SLAVE = "slave"
