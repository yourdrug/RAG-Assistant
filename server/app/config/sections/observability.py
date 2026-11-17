"""Runtime controls for answer traces and quality log retention."""

from pydantic import BaseModel, PositiveInt


class ObservabilitySection(BaseModel):
    trace_enabled: bool = True
    chat_trace_retention_days: PositiveInt = 30
    chat_log_retention_days: PositiveInt = 180
