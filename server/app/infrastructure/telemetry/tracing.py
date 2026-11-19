"""OpenTelemetry tracing: provider setup, cross-process context and paid model-call accounting.

Spans never carry prompts, answers, document text or credentials: only models, operations,
token counts, cost and identifiers. Every exported span passes ``PrivacySpanExporter``: URL
query strings, exception messages/stack traces and status descriptions are removed, since
search parameters and provider errors may quote private input. Errors keep only their type.

Without ``otel_enabled`` every helper here is a no-op: no context, no usage parsing.

Paid model calls (LLM, embeddings, reranking) are CLIENT spans named ``<operation> <model>``
with ``rag.paid_call=true``. Every enclosing step span and the request/job root span carry a
tally of the logical calls made beneath it: ``rag.paid_calls.total``, ``rag.paid_calls.<operation>``,
``rag.paid_calls.attempts`` (live provider responses, retries included), ``rag.paid_calls.replayed``
and summed known tokens/cost with ``rag.paid_calls.cost_complete``.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
import logging
import math
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlsplit

from opentelemetry import context as otel_context, propagate, trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.utils import suppress_instrumentation
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import Event, ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.trace import INVALID_SPAN, SpanKind, Status, StatusCode


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Callable, Coroutine, Iterator, Sequence

    from opentelemetry.context import Context
    from opentelemetry.sdk.trace.export import SpanExportResult
    from opentelemetry.trace import Span
    from opentelemetry.util.types import Attributes


__all__ = [
    "PaidCall",
    "PaidCallTally",
    "PrivacySpanExporter",
    "annotate_current_span",
    "configure_tracing",
    "context_from_carrier",
    "current_trace_carrier",
    "paid_call",
    "paid_call_active",
    "paid_call_scope",
    "record_paid_usage",
    "shutdown_tracing",
    "step_span",
    "suppress_instrumentation",
    "traced_step",
    "traced_stream",
    "tracing_enabled",
]

log = logging.getLogger("default")

TRACER = trace.get_tracer("rag-assistant")
TALLIES_KEY = otel_context.create_key("rag.paid_call_tallies")
CALL_KEY = otel_context.create_key("rag.paid_call")

# Old and current HTTP semantic conventions; values keep scheme, host and path only.
URL_ATTRIBUTES = frozenset({"http.url", "http.target", "url.full", "url.original", "http.request.uri"})
DROPPED_ATTRIBUTES = frozenset({"url.query", "exception.message", "exception.stacktrace"})


# ---------------------------------------------------------------------------
# Export privacy filter
# ---------------------------------------------------------------------------


def without_query(value: str) -> str:
    return value.split("#", 1)[0].split("?", 1)[0]


def safe_attributes(attributes: Attributes) -> dict[str, Any]:
    result: dict[str, Any] = {}

    for key, value in (attributes or {}).items():
        if key in DROPPED_ATTRIBUTES:
            continue

        result[key] = without_query(value) if key in URL_ATTRIBUTES and isinstance(value, str) else value

    return result


def safe_span(span: ReadableSpan) -> ReadableSpan:
    return ReadableSpan(
        name=span.name,
        context=span.context,
        parent=span.parent,
        resource=span.resource,
        attributes=safe_attributes(span.attributes),
        events=[
            Event(event.name, safe_attributes(event.attributes), event.timestamp) for event in span.events
        ],
        links=span.links,
        kind=span.kind,
        # The description is usually the exception text; the code and error.type remain.
        status=Status(span.status.status_code),
        start_time=span.start_time,
        end_time=span.end_time,
        instrumentation_scope=span.instrumentation_scope,
    )


class PrivacySpanExporter(SpanExporter):
    """Strip query strings and error texts from every span before it leaves the process."""

    def __init__(self, exporter: SpanExporter) -> None:
        self.exporter = exporter

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return self.exporter.export([safe_span(span) for span in spans])

    def shutdown(self) -> None:
        self.exporter.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self.exporter.force_flush(timeout_millis)


# ---------------------------------------------------------------------------
# Process setup
# ---------------------------------------------------------------------------


@dataclass
class TracingState:
    provider: TracerProvider | None = None
    enabled: bool = False


state = TracingState()


def configure_tracing(
    service_name: str, *, enabled: bool, endpoint: str = "", exporter: SpanExporter | None = None
) -> bool:
    """Enable tracing for this process; safe to repeat after ``shutdown_tracing``.

    The global provider can be installed only once per process, so it is kept for the process
    lifetime (flushed at exit by the SDK) and a repeated lifespan re-enables it as it was.
    """
    if not enabled:
        return state.enabled

    if state.provider is None:
        provider = TracerProvider(resource=Resource.create({SERVICE_NAME: service_name}))
        target = exporter or (
            OTLPSpanExporter(endpoint=f"{endpoint.rstrip('/')}/v1/traces") if endpoint else OTLPSpanExporter()
        )
        provider.add_span_processor(BatchSpanProcessor(PrivacySpanExporter(target)))
        trace.set_tracer_provider(provider)

        if trace.get_tracer_provider() is not provider:
            log.warning("Another tracer provider is already installed; tracing stays disabled")
            return False

        state.provider = provider

    if not state.enabled:
        # Patches the httpx transports: Qdrant REST, the OpenAI SDK and the inference pools.
        HTTPXClientInstrumentor().instrument(tracer_provider=state.provider)
        state.enabled = True
        log.info("OpenTelemetry tracing enabled (service=%s)", service_name)

    return True


def shutdown_tracing() -> None:
    """Disable tracing and flush pending spans; the provider stays installed for a later restart."""
    if state.enabled:
        HTTPXClientInstrumentor().uninstrument()
        state.enabled = False

    if state.provider is not None:
        state.provider.force_flush()


def tracing_enabled() -> bool:
    return state.enabled


def current_trace_carrier() -> dict[str, str]:
    """W3C trace context of the current span, for state that crosses processes."""
    carrier: dict[str, str] = {}

    if state.enabled:
        propagate.inject(carrier)

    return carrier


def context_from_carrier(carrier: dict[str, str] | None) -> Context | None:
    return propagate.extract(carrier) if carrier else None


def annotate_current_span(**attributes: str | float | bool) -> None:
    span = trace.get_current_span()

    if span.is_recording():
        span.set_attributes(attributes)


@contextmanager
def error_type_status(span: Span) -> Iterator[None]:
    """Mark a failed span with the exception type only; its text may quote private input."""
    try:
        yield
    except Exception as exc:
        if span.is_recording():
            span.set_attribute("error.type", type(exc).__qualname__)
            span.set_status(Status(StatusCode.ERROR))

        raise


# ---------------------------------------------------------------------------
# Paid-call accounting
# ---------------------------------------------------------------------------


@dataclass
class PaidCall:
    """One logical model call; a retrying client may receive several provider responses."""

    operation: str
    live_attempts: int = 0
    replayed_attempts: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    cost_missing: bool = False

    @property
    def replayed(self) -> bool:
        """Every response came from a recording: nothing was paid for this call now."""
        return self.live_attempts == 0 and self.replayed_attempts > 0

    @property
    def cost_complete(self) -> bool:
        # No usage at all (failure, stream without usage) still means a request was billed.
        return self.live_attempts > 0 and not self.cost_missing

    def add_attempt(
        self,
        *,
        replayed: bool,
        input_tokens: int | None,
        output_tokens: int | None,
        cost_usd: float | None,
    ) -> None:
        if replayed:
            self.replayed_attempts += 1
            return

        self.live_attempts += 1

        if input_tokens is not None:
            self.input_tokens = (self.input_tokens or 0) + input_tokens

        if output_tokens is not None:
            self.output_tokens = (self.output_tokens or 0) + output_tokens

        if cost_usd is None:
            self.cost_missing = True
        else:
            self.cost_usd = (self.cost_usd or 0.0) + cost_usd


@dataclass
class PaidCallTally:
    calls: Counter[str] = field(default_factory=Counter)
    attempts: int = 0
    replayed: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    known_cost_usd: float = 0.0
    cost_complete: bool = True

    def add(self, call: PaidCall) -> None:
        if call.replayed:
            self.replayed += 1
            return

        self.calls[call.operation] += 1
        self.attempts += max(call.live_attempts, 1)
        self.input_tokens += call.input_tokens or 0
        self.output_tokens += call.output_tokens or 0
        self.known_cost_usd += call.cost_usd or 0.0
        self.cost_complete = self.cost_complete and call.cost_complete

    def attributes(self) -> dict[str, str | int | float | bool]:
        total = sum(self.calls.values())
        attributes: dict[str, str | int | float | bool] = {
            "rag.paid_calls.total": total,
            **{f"rag.paid_calls.{operation}": count for operation, count in self.calls.items()},
        }

        if self.replayed:
            attributes["rag.paid_calls.replayed"] = self.replayed

        if total:
            attributes.update(
                {
                    "rag.paid_calls.attempts": self.attempts,
                    "rag.paid_calls.input_tokens": self.input_tokens,
                    "rag.paid_calls.output_tokens": self.output_tokens,
                    "rag.paid_calls.known_cost_usd": self.known_cost_usd,
                    "rag.paid_calls.cost_complete": self.cost_complete,
                }
            )

        return attributes


@contextmanager
def paid_call_scope(span: Span | None = None) -> Iterator[PaidCallTally]:
    """Count paid calls made inside and write the tally onto ``span`` (default: current span)."""
    tally = PaidCallTally()

    if not state.enabled:
        yield tally
        return

    target = span or trace.get_current_span()
    parents = cast("tuple[PaidCallTally, ...]", otel_context.get_value(TALLIES_KEY) or ())
    token = otel_context.attach(otel_context.set_value(TALLIES_KEY, (*parents, tally)))
    try:
        yield tally
    finally:
        otel_context.detach(token)

        if target.is_recording():
            target.set_attributes(tally.attributes())


@contextmanager
def step_span(
    name: str,
    *,
    kind: SpanKind = SpanKind.INTERNAL,
    parent: Context | None = None,
    attributes: dict[str, str | int | float | bool] | None = None,
) -> Iterator[Span]:
    """A span that also tallies the paid calls made beneath it.

    With a remote ``parent`` the span's context derives from that parent alone, so the tally
    chain restarts: callers' tallies live in another process.
    """
    if not state.enabled:
        yield INVALID_SPAN
        return

    with (
        TRACER.start_as_current_span(
            name,
            context=parent,
            kind=kind,
            attributes=attributes,
            record_exception=False,
            set_status_on_exception=False,
        ) as span,
        error_type_status(span),
        paid_call_scope(span),
    ):
        yield span


def traced_step[**P, R](
    name: str,
) -> Callable[[Callable[P, Coroutine[Any, Any, R]]], Callable[P, Coroutine[Any, Any, R]]]:
    def decorate(function: Callable[P, Coroutine[Any, Any, R]]) -> Callable[P, Coroutine[Any, Any, R]]:
        @wraps(function)
        async def run(*args: P.args, **kwargs: P.kwargs) -> R:
            with step_span(name):
                return await function(*args, **kwargs)

        return run

    return decorate


def traced_stream[**P, Y](
    name: str,
) -> Callable[[Callable[P, AsyncIterator[Y]]], Callable[P, AsyncGenerator[Y]]]:
    def decorate(function: Callable[P, AsyncIterator[Y]]) -> Callable[P, AsyncGenerator[Y]]:
        @wraps(function)
        async def run(*args: P.args, **kwargs: P.kwargs) -> AsyncGenerator[Y]:
            with step_span(name):
                stream = function(*args, **kwargs)
                try:
                    async for item in stream:
                        yield item
                finally:
                    aclose = getattr(stream, "aclose", None)

                    if aclose is not None:
                        await aclose()

        return run

    return decorate


def endpoint_address(base_url: str) -> str:
    return urlsplit(base_url).hostname or ""


@contextmanager
def paid_call(operation: str, model: str, *, base_url: str = "", **attributes: Any) -> Iterator[PaidCall]:  # noqa: ANN401
    """CLIENT span of one billable model call; usage is filled by ``record_paid_usage``."""
    call = PaidCall(operation)

    if not state.enabled:
        yield call
        return

    tallies = cast("tuple[PaidCallTally, ...]", otel_context.get_value(TALLIES_KEY) or ())
    span_attributes: dict[str, Any] = {
        "gen_ai.operation.name": operation,
        "gen_ai.request.model": model,
        "rag.paid_call": True,
        **attributes,
    }

    if address := endpoint_address(base_url):
        span_attributes["server.address"] = address

    with (
        TRACER.start_as_current_span(
            f"{operation} {model}".strip(),
            kind=SpanKind.CLIENT,
            record_exception=False,
            set_status_on_exception=False,
        ) as span,
        error_type_status(span),
    ):
        span.set_attributes(span_attributes)
        token = otel_context.attach(otel_context.set_value(CALL_KEY, call))
        try:
            yield call
        finally:
            otel_context.detach(token)
            span.set_attributes(
                {
                    "rag.paid_call.attempts": call.live_attempts,
                    "rag.paid_call.replayed": call.replayed,
                    "rag.paid_call.cost_complete": call.cost_complete,
                }
            )

            for tally in tallies:
                tally.add(call)


def paid_call_active() -> bool:
    return state.enabled and otel_context.get_value(CALL_KEY) is not None


def finite_number(value: object) -> float | None:
    """Provider usage fields arrive as numbers, numeric strings or garbage."""
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        return None

    try:
        number = float(value)
    except ValueError:
        return None

    return number if math.isfinite(number) and number >= 0 else None


def token_count(value: object) -> int | None:
    number = finite_number(value)
    return None if number is None else int(number)


def record_paid_usage(
    *,
    input_tokens: object,
    output_tokens: object,
    cost_usd: object,
    replayed: bool = False,
    response_model: object = None,
) -> None:
    """Attach one provider response's usage to the enclosing ``paid_call``.

    Telemetry never interrupts response handling: unparsable values count as unknown.
    """
    call = cast("PaidCall | None", otel_context.get_value(CALL_KEY)) if state.enabled else None

    if call is None:
        return

    try:
        call.add_attempt(
            replayed=replayed,
            input_tokens=token_count(input_tokens),
            output_tokens=token_count(output_tokens),
            cost_usd=finite_number(cost_usd),
        )
        span = trace.get_current_span()

        if not span.is_recording():
            return

        usage: dict[str, str | int | float | bool] = {}

        if call.input_tokens is not None:
            usage["gen_ai.usage.input_tokens"] = call.input_tokens

        if call.output_tokens is not None:
            usage["gen_ai.usage.output_tokens"] = call.output_tokens

        if call.cost_usd is not None:
            usage["rag.cost_usd"] = call.cost_usd

        if isinstance(response_model, str) and response_model:
            usage["gen_ai.response.model"] = response_model

        span.set_attributes(usage)
    except Exception as exc:  # noqa: BLE001 — accounting must not fail a successful model call
        log.warning("Model usage not attributed to the trace: %s", type(exc).__name__)
