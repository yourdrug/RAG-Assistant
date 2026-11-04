"""Request-local judge usage, including successful responses rejected by validation."""

from contextvars import ContextVar
from typing import Any

from infrastructure.ml.usage_capture import active_llm_usage, record_model_response


judge_context: ContextVar[dict[str, Any] | None] = ContextVar("benchmark_judge_context", default=None)

judge_usage: ContextVar[list[dict[str, Any]] | None] = ContextVar("benchmark_judge_usage", default=None)


def record_judge_usage(response: object) -> None:
    records = judge_usage.get()
    usage = getattr(response, "usage", None)

    if records is not None and usage is not None:
        captured: list[dict[str, Any]] = []
        token = active_llm_usage.set(captured)
        try:
            record_model_response(response, operation="judge")
        finally:
            active_llm_usage.reset(token)
        records.extend(captured)
