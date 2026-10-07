"""Request-local judge usage, including successful responses rejected by validation."""

from contextvars import ContextVar

judge_usage: ContextVar[list[dict] | None] = ContextVar("benchmark_judge_usage", default=None)


def record_judge_usage(response) -> None:
    records = judge_usage.get()
    usage = getattr(response, "usage", None)
    if records is not None and usage is not None:
        records.append({"input_tokens": usage.prompt_tokens, "output_tokens": usage.completion_tokens})
