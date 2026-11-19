"""Runtime controls for answer traces and quality log retention; static OpenTelemetry export."""

from pydantic import BaseModel, PositiveInt

from config.ownership import Static


class ObservabilitySection(BaseModel):
    trace_enabled: bool = True
    chat_trace_retention_days: PositiveInt = 30
    chat_log_retention_days: PositiveInt = 180
    # Distributed tracing (OpenTelemetry, OTLP/HTTP). Empty endpoint: OTEL_EXPORTER_OTLP_* env or SDK default.
    # Sampling follows OTEL_TRACES_SAMPLER/OTEL_TRACES_SAMPLER_ARG (default: parent-based, always on).
    otel_enabled: Static[bool] = False
    otel_exporter_endpoint: Static[str] = ""
