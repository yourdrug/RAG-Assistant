"""Request-local judge usage, including successful responses rejected by validation."""

from contextvars import ContextVar
from infrastructure.ml.usage_capture import active_llm_usage, record_model_response

judge_context: ContextVar[dict | None] = ContextVar("benchmark_judge_context", default=None)

judge_usage: ContextVar[list[dict] | None] = ContextVar("benchmark_judge_usage", default=None)


def record_judge_usage(response) -> None:
    records = judge_usage.get()
    usage = getattr(response, "usage", None)
    if records is not None and usage is not None:
        captured: list[dict] = []
        token = active_llm_usage.set(captured)
        try:
            record_model_response(response, operation='judge')
        finally:
            active_llm_usage.reset(token)
        records.extend(captured)
