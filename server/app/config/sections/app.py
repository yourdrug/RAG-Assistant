"""Application metadata, stage and timezone."""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from pydantic import BaseModel

from application.runtime.enums import AppStage
from config.ownership import Metadata, Static


class AppSection(BaseModel):
    # ── Timezone ────────────────────────────────────────────────────────────
    timezone: Static[str] = "UTC"

    # ── App metadata ────────────────────────────────────────────────────────
    version: Metadata[str] = ""
    service_start_datetime: Metadata[str] = ""
    stage: Static[AppStage] = AppStage.DEVELOPMENT

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
