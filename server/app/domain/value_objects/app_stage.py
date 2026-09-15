"""Application deployment stage."""

from __future__ import annotations

from enum import StrEnum


class AppStage(StrEnum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PROD = "prod"
