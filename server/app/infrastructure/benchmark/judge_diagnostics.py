"""Safe judge telemetry: correlation and response structure, never document content."""

import hashlib
import json
import logging

from pydantic import ValidationError

from infrastructure.benchmark.token_usage import judge_context

logger = logging.getLogger("default")


class JudgeResponseError(ValueError):
    def __init__(self, code: str, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


def response_metadata(response) -> dict:
    choice = response.choices[0] if response.choices else None
    usage = getattr(response, "usage", None)
    details = getattr(usage, "completion_tokens_details", None)
    message = getattr(choice, "message", None)
    content = getattr(message, "content", None) or ""
    return {
        "response_id": getattr(response, "id", None),
        "provider_request_id": getattr(response, "_request_id", None),
        "response_model": getattr(response, "model", None),
        "finish_reason": getattr(choice, "finish_reason", None),
        "content_chars": len(content),
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "input_tokens": getattr(usage, "prompt_tokens", None),
        "output_tokens": getattr(usage, "completion_tokens", None),
        "reasoning_tokens": getattr(details, "reasoning_tokens", None),
        "reasoning_chars": len(getattr(message, "reasoning", None) or ""),
        "provider_refusal": bool(getattr(message, "refusal", None)),
    }


def failure_metadata(exc: Exception) -> dict:
    response = getattr(exc, "response", None)
    return {
        **(exc.details if isinstance(exc, JudgeResponseError) else {}),
        "error_code": getattr(exc, "code", None)
        if isinstance(exc, JudgeResponseError)
        else type(exc).__name__,
        "error_type": type(exc).__name__,
        "http_status": getattr(exc, "status_code", None),
        "provider_request_id": getattr(exc, "request_id", None),
        "retry_after": getattr(response, "headers", {}).get("retry-after") if response is not None else None,
    }


def log_judge_event(event: dict, *, warning: bool = False) -> None:
    record = {**(judge_context.get() or {}), **event}
    logger.log(
        logging.WARNING if warning else logging.INFO,
        "Benchmark judge %s",
        json.dumps(record, ensure_ascii=False, sort_keys=True),
        extra={"benchmark_judge": record},
    )


def metric_error_codes(exc: Exception) -> list[str]:
    if isinstance(exc, ValidationError):
        return [error["type"] for error in exc.errors(include_input=False, include_url=False)]
    return ["missing_or_invalid_metric"]
