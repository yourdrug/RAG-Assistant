"""Request-local model usage; unknown cost remains unknown."""

from __future__ import annotations

from contextvars import ContextVar
import logging
from typing import TYPE_CHECKING, Any, Protocol, Self, cast

from infrastructure.ml.replay.recording import REPLAYED_USAGE_FLAG
from infrastructure.telemetry.tracing import paid_call_active, record_paid_usage


log = logging.getLogger("default")


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable


active_llm_usage: ContextVar[list[dict[str, Any]] | None] = ContextVar("rag_model_usage", default=None)
active_stream_response: ContextVar[dict[str, Any] | None] = ContextVar("rag_stream_response", default=None)


def extract_usage_from_langchain(response: object) -> tuple[int | None, int | None]:
    """Extract (input_tokens, output_tokens) from a LangChain AIMessage or chunk.

    Supports both usage_metadata (LangChain v0.3+) and response_metadata.
    Returns (None, None) if not available.
    """
    # Try usage_metadata first (LangChain >=0.3)
    usage = getattr(response, "usage_metadata", None)

    if usage and isinstance(usage, dict):
        return usage.get("input_tokens"), usage.get("output_tokens")

    # Try response_metadata (older LangChain / OpenAI format)
    meta = getattr(response, "response_metadata", None)

    if meta and isinstance(meta, dict):
        usage = meta.get("usage", {})

        if usage:
            return usage.get("prompt_tokens"), usage.get("completion_tokens")

    return None, None


class RawCompletionStream(Protocol):
    async def __aenter__(self) -> object: ...
    async def __aexit__(self, *args: object) -> bool | None: ...
    def __aiter__(self) -> AsyncIterator[object]: ...


class RawCompletions(Protocol):
    async def create(self, **kwargs: object) -> object: ...


class UsageCapturingStream:
    """Preserve raw OpenAI-compatible usage discarded by LangChain conversion."""

    def __init__(self, stream: object) -> None:
        self.stream = cast("RawCompletionStream", stream)

    def __getattr__(self, name: str) -> object:
        return getattr(self.stream, name)

    async def __aenter__(self) -> Self:
        await self.stream.__aenter__()
        return self

    async def __aexit__(self, *args: object) -> bool | None:
        return await self.stream.__aexit__(*args)

    async def __aiter__(self) -> AsyncIterator[object]:
        async for chunk in self.stream:
            captured = active_stream_response.get()

            if captured is not None:
                for name in ("id", "model", "usage"):
                    value = getattr(chunk, name, None)

                    if value is not None:
                        captured[name] = value

            yield chunk


class UsageCapturingCompletions:
    def __init__(self, client: object) -> None:
        self.client = cast("RawCompletions", client)

    def __getattr__(self, name: str) -> object:
        return getattr(self.client, name)

    async def create(self, **kwargs: object) -> object:
        response = await self.client.create(**kwargs)

        if kwargs.get("stream") and active_stream_response.get() is not None:
            return UsageCapturingStream(response)

        return response


def record_model_response(response: object, *, model: str = "", operation: str = "llm") -> None:
    if active_llm_usage.get() is not None:
        capture_model_response(response, model=model, operation=operation)
    elif paid_call_active():
        trace_only(capture_model_response, response, model=model, operation=operation)


def trace_only[**P](capture: Callable[P, None], *args: P.args, **kwargs: P.kwargs) -> None:
    """Usage parsed only for the trace must never fail the response that carried it."""
    try:
        capture(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 — telemetry-only parsing of provider metadata
        log.warning("Model usage not attributed to the trace: %s", type(exc).__name__)


def capture_model_response(response: object, *, model: str, operation: str) -> None:
    records = active_llm_usage.get()
    metadata = getattr(response, "response_metadata", {}) or {}
    raw_usage = getattr(response, "usage", None)
    input_tokens, output_tokens = extract_usage_from_langchain(response)
    usage = (
        raw_usage.model_dump() if raw_usage is not None and hasattr(raw_usage, "model_dump") else raw_usage
    )
    usage = usage or metadata.get("token_usage") or metadata.get("usage") or {}

    if not isinstance(usage, dict):
        usage = {key: getattr(usage, key, None) for key in ("prompt_tokens", "completion_tokens", "cost")}

    input_tokens = input_tokens if input_tokens is not None else usage.get("prompt_tokens")
    output_tokens = output_tokens if output_tokens is not None else usage.get("completion_tokens")
    actual_model = (
        getattr(response, "model", None) or metadata.get("model_name") or metadata.get("model") or model
    )
    response_id = metadata.get("id") or getattr(response, "id", None)
    record = {
        "model": actual_model,
        "operation": operation,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": usage.get("cost"),
        "response_id": response_id
        if isinstance(response_id, str) and response_id.startswith("gen-")
        else None,
    }

    replayed = usage.get(REPLAYED_USAGE_FLAG) is True
    record_paid_usage(
        input_tokens=0 if replayed else input_tokens,
        output_tokens=0 if replayed else output_tokens,
        cost_usd=0.0 if replayed else usage.get("cost"),
        replayed=replayed,
        response_model=actual_model if isinstance(actual_model, str) else None,
    )

    if records is None:
        return

    if replayed:
        # A recorded response was reused: nothing was paid now, its original usage is informational.
        record.update(
            replayed=True,
            replayed_input_tokens=input_tokens,
            replayed_output_tokens=output_tokens,
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
        )

    records.append(record)


def summarize_usage(records: list[dict[str, Any]]) -> dict[str, Any]:
    complete = all(record.get("cost_usd") is not None for record in records)
    known = sum(float(record["cost_usd"]) for record in records if record.get("cost_usd") is not None)
    return {
        "calls": len(records),
        "replayed_calls": sum(1 for record in records if record.get("replayed")),
        "input_tokens": sum(record.get("input_tokens") or 0 for record in records),
        "output_tokens": sum(record.get("output_tokens") or 0 for record in records),
        "tokens_complete": all(
            record.get("input_tokens") is not None and record.get("output_tokens") is not None
            for record in records
        ),
        "known_cost_usd": known,
        "cost_usd": known if complete else None,
        "cost_complete": complete,
    }


def record_inference_response(data: dict[str, Any], *, model: str, operation: str) -> None:
    if active_llm_usage.get() is not None:
        capture_inference_response(data, model=model, operation=operation)
    elif paid_call_active():
        trace_only(capture_inference_response, data, model=model, operation=operation)


def capture_inference_response(data: dict[str, Any], *, model: str, operation: str) -> None:
    records = active_llm_usage.get()
    usage = data.get("usage") or {}
    status = data.get("inference_status") or {}
    input_tokens = data.get("input_tokens") if operation == "reranking" else usage.get("prompt_tokens")
    cost_usd = usage.get("cost", status.get("cost"))
    record_paid_usage(input_tokens=input_tokens, output_tokens=0, cost_usd=cost_usd)

    if records is None:
        return

    records.append(
        {
            "model": data.get("model") or model,
            "operation": operation,
            "input_tokens": input_tokens,
            "output_tokens": 0,
            "cost_usd": cost_usd,
            "response_id": data.get("request_id"),
        }
    )
