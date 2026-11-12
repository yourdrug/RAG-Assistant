"""Application metadata, stage and timezone."""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, field_validator

from application.runtime.enums import AppStage
from config.ownership import Metadata, Static


class AppSection(BaseModel):
    # ── Timezone ────────────────────────────────────────────────────────────
    timezone: Static[str] = "UTC"

    # ── App metadata ────────────────────────────────────────────────────────
    version: Metadata[str] = ""
    service_start_datetime: Metadata[str] = ""
    stage: Static[AppStage] = AppStage.DEVELOPMENT

    @field_validator("timezone")
    @classmethod
    def check_timezone(cls, value: str) -> str:
        # Fail at startup, not on the first ``tz`` access at runtime.
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            message = f"Unknown TIMEZONE {value!r}"
            raise ValueError(message) from exc

        return value

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    @property
    def uptime_seconds(self) -> float:
        if not self.service_start_datetime:
            return 0.0

        try:
            start = datetime.fromisoformat(self.service_start_datetime)
            now = datetime.now(tz=UTC)
            return (now - start).total_seconds()
        except ValueError:
            return 0.0
