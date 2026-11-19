"""OpenTelemetry tracing: privacy of exported spans, paid-call accounting and API → worker links."""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import httpx
from opentelemetry.instrumentation.utils import is_instrumentation_enabled
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from domain.runtime.jobs import BackgroundJob, JobType
from infrastructure.database.models import BackgroundJobModel
from infrastructure.ml.clients.inference_clients import EmbeddingsClient
from infrastructure.ml.usage_capture import record_model_response
from infrastructure.repositories.misc.sqlalchemy_background_job_repository import (
    SQLAlchemyBackgroundJobRepository,
)
from infrastructure.telemetry import tracing
from infrastructure.telemetry.asgi import instrument_api
from infrastructure.telemetry.tracing import (
    configure_tracing,
    paid_call,
    record_paid_usage,
    shutdown_tracing,
    step_span,
    traced_step,
    traced_stream,
)
from infrastructure.worker import runtime
from infrastructure.worker.registry import TASKS


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from opentelemetry.sdk.trace import ReadableSpan
    from sqlalchemy.ext.asyncio import AsyncSession


EXPORTER = InMemorySpanExporter()
MARKER = "конфиденциальный-маркер"


@pytest.fixture(scope="module", autouse=True)
def tracing_on() -> Iterator[None]:
    # One provider per process: other modules see tracing disabled again after this one.
    assert configure_tracing("test", enabled=True, exporter=EXPORTER)
    yield
    shutdown_tracing()


@pytest.fixture(autouse=True)
def spans(tracing_on: None) -> Iterator[None]:
    del tracing_on
    configure_tracing("test", enabled=True)
    finished()
    EXPORTER.clear()
    yield
    EXPORTER.clear()


def finished() -> tuple[ReadableSpan, ...]:
    provider = tracing.state.provider
    assert provider is not None
    provider.force_flush()
    return EXPORTER.get_finished_spans()


def by_name(name: str) -> ReadableSpan:
    matches = [span for span in finished() if span.name == name]
    assert len(matches) == 1, [span.name for span in finished()]
    return matches[0]


def attrs(span: ReadableSpan) -> dict[str, Any]:
    return dict(span.attributes or {})


def exported_text() -> str:
    """Everything an exporter would ship for the finished spans."""
    return repr(
        [
            (
                span.name,
                attrs(span),
                span.status.description,
                [(event.name, dict(event.attributes or {})) for event in span.events],
            )
            for span in finished()
        ]
    )


@traced_step("rag.condense")
async def condense() -> None:
    with paid_call("chat", "aux-model"):
        record_paid_usage(input_tokens=10, output_tokens=2, cost_usd=0.001)

    with paid_call("chat", "aux-model"):
        record_paid_usage(input_tokens=5, output_tokens=1, cost_usd=None)


@traced_step("rag.retrieve")
async def retrieve() -> None:
    with paid_call("embeddings", "embed-model"):
        record_paid_usage(input_tokens=7, output_tokens=0, cost_usd=0.0001)


@pytest.mark.asyncio
async def test_paid_calls_are_tallied_on_each_step_and_on_the_request() -> None:
    with step_span("rag.pipeline"):
        await condense()
        await retrieve()

    pipeline = attrs(by_name("rag.pipeline"))
    assert pipeline["rag.paid_calls.total"] == 3
    assert pipeline["rag.paid_calls.chat"] == 2
    assert pipeline["rag.paid_calls.embeddings"] == 1
    assert pipeline["rag.paid_calls.input_tokens"] == 22
    # One call reported no cost: the sum is a lower bound, flagged as incomplete.
    assert pipeline["rag.paid_calls.cost_complete"] is False

    condense_span = by_name("rag.condense")
    assert attrs(condense_span)["rag.paid_calls.total"] == 2
    assert "rag.paid_calls.embeddings" not in attrs(condense_span)
    assert attrs(by_name("rag.retrieve"))["rag.paid_calls.total"] == 1

    chat_calls = [span for span in finished() if span.name == "chat aux-model"]
    assert len(chat_calls) == 2
    assert all(span.kind == SpanKind.CLIENT for span in chat_calls)
    assert {span.parent.span_id for span in chat_calls if span.parent} == {condense_span.context.span_id}
    assert attrs(chat_calls[0])["gen_ai.usage.input_tokens"] == 10
    assert attrs(chat_calls[0])["rag.paid_call"] is True


@pytest.mark.asyncio
async def test_a_step_without_paid_calls_reports_zero() -> None:
    @traced_step("rag.input_safety")
    async def scan() -> None:
        return None

    await scan()

    assert attrs(by_name("rag.input_safety"))["rag.paid_calls.total"] == 0


def test_replayed_llm_response_is_visible_but_not_counted_as_paid() -> None:
    usage = {"prompt_tokens": 100, "completion_tokens": 20, "cost": 0.5, "replayed": True}
    response = SimpleNamespace(usage=SimpleNamespace(model_dump=lambda: usage), model="judge-model")

    with step_span("job benchmark"):
        with paid_call("chat", "judge-model"):
            record_model_response(response, operation="judge")
        with paid_call("chat", "judge-model"):
            record_paid_usage(input_tokens=1, output_tokens=1, cost_usd=0.01)

    job = attrs(by_name("job benchmark"))
    assert job["rag.paid_calls.total"] == 1
    assert job["rag.paid_calls.replayed"] == 1
    assert job["rag.paid_calls.input_tokens"] == 1


def test_retry_without_cost_makes_the_logical_call_cost_incomplete() -> None:
    with step_span("rag.evidence_gate"), paid_call("chat", "aux-model"):
        record_paid_usage(input_tokens=10, output_tokens=1, cost_usd=0.01)
        record_paid_usage(input_tokens=12, output_tokens=1, cost_usd=None)

    step = attrs(by_name("rag.evidence_gate"))
    assert step["rag.paid_calls.total"] == 1
    assert step["rag.paid_calls.attempts"] == 2
    assert step["rag.paid_calls.input_tokens"] == 22
    assert step["rag.paid_calls.known_cost_usd"] == pytest.approx(0.01)
    assert step["rag.paid_calls.cost_complete"] is False


def test_live_attempt_followed_by_replay_stays_paid() -> None:
    with step_span("job benchmark"), paid_call("chat", "judge-model"):
        record_paid_usage(input_tokens=10, output_tokens=1, cost_usd=0.02)
        record_paid_usage(input_tokens=0, output_tokens=0, cost_usd=0.0, replayed=True)

    job = attrs(by_name("job benchmark"))
    assert job["rag.paid_calls.total"] == 1
    assert job["rag.paid_calls.attempts"] == 1
    assert job["rag.paid_calls.known_cost_usd"] == pytest.approx(0.02)
    assert job["rag.paid_calls.cost_complete"] is True
    assert "rag.paid_calls.replayed" not in job


def test_call_without_any_usage_is_paid_with_unknown_cost() -> None:
    with step_span("rag.generate"), paid_call("chat", "main-model"):
        pass

    step = attrs(by_name("rag.generate"))
    assert step["rag.paid_calls.total"] == 1
    assert step["rag.paid_calls.cost_complete"] is False


@pytest.mark.asyncio
async def test_streamed_step_counts_calls_made_while_streaming_and_closes_early() -> None:
    @traced_stream("rag.generate")
    async def stream() -> AsyncIterator[str]:
        with paid_call("chat", "main-model"):
            yield "a"
            yield "b"

    parts = stream()
    assert await anext(parts) == "a"
    await parts.aclose()

    assert attrs(by_name("rag.generate"))["rag.paid_calls.chat"] == 1


def embeddings_client(usage: dict[str, object]) -> EmbeddingsClient:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1, 0.2]}], "usage": usage})

    client = EmbeddingsClient("https://embed.example/v1", "bge-m3", api_key="sk-test")
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client


@pytest.mark.asyncio
async def test_embedding_span_carries_usage_but_never_the_input_text() -> None:
    client = embeddings_client({"prompt_tokens": 9, "cost": "0.00002"})
    try:
        assert await client.embed_documents([MARKER]) == [[0.1, 0.2]]
    finally:
        await client.close()

    values = attrs(by_name("embeddings bge-m3"))
    assert values["gen_ai.usage.input_tokens"] == 9
    assert values["rag.cost_usd"] == pytest.approx(0.00002)
    assert values["rag.batch_size"] == 1
    assert values["server.address"] == "embed.example"
    assert MARKER not in exported_text()
    assert "sk-test" not in exported_text()


@pytest.mark.asyncio
@pytest.mark.parametrize("cost", ["0.01", "not-a-number", {"amount": 1}, "nan"])
async def test_unparsable_usage_never_breaks_a_successful_call(cost: object) -> None:
    client = embeddings_client({"prompt_tokens": "9", "cost": cost})
    try:
        assert await client.embed_query("q") == [0.1, 0.2]
    finally:
        await client.close()

    assert attrs(by_name("embeddings bge-m3"))["rag.paid_call.attempts"] == 1


@pytest.mark.asyncio
async def test_disabled_tracing_skips_telemetry_entirely() -> None:
    shutdown_tracing()
    client = embeddings_client({"prompt_tokens": 9, "cost": "0.01"})
    try:
        with step_span("rag.pipeline") as span:
            assert not span.is_recording()
            assert await client.embed_query("q") == [0.1, 0.2]
            assert not tracing.paid_call_active()
    finally:
        await client.close()

    assert finished() == ()


def test_restarted_lifespan_exports_again() -> None:
    shutdown_tracing()
    assert configure_tracing("test", enabled=True)
    shutdown_tracing()
    assert configure_tracing("test", enabled=True)

    with step_span("after restart"):
        pass

    assert by_name("after restart").name == "after restart"


def test_failed_paid_call_exports_error_type_without_its_text() -> None:
    def fail() -> None:
        message = f"provider echoed: {MARKER}"
        raise ValueError(message)

    with pytest.raises(ValueError, match=MARKER), paid_call("chat", "main-model"):
        fail()

    span = by_name("chat main-model")
    assert span.status.status_code == StatusCode.ERROR
    assert attrs(span)["error.type"] == "ValueError"
    assert MARKER not in exported_text()


@pytest.mark.asyncio
async def test_server_spans_drop_query_strings_and_exception_texts() -> None:
    app = FastAPI()

    @app.get("/admin/chat-logs")
    async def chat_logs(search: str) -> dict[str, str]:
        del search
        return {}

    @app.get("/boom")
    async def boom() -> dict[str, str]:
        message = f"failed for {MARKER}"
        raise RuntimeError(message)

    instrument_api(app)
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)  # type: ignore[arg-type]  # Starlette and HTTPX disagree on the ASGI send annotation.
    async with httpx.AsyncClient(transport=transport, base_url="http://api") as client:
        assert (await client.get("/admin/chat-logs", params={"search": MARKER})).status_code == 200
        assert (await client.get("/boom")).status_code == 500

    by_name("GET /admin/chat-logs")  # asserts exactly one matching span
    assert by_name("GET /boom").status.status_code == StatusCode.ERROR
    text = exported_text()
    assert MARKER not in text
    assert "search=" not in text
    assert "%D0%BA" not in text.upper()  # the percent-encoded marker


@pytest.mark.asyncio
async def test_api_server_span_reports_paid_calls_of_a_streamed_response() -> None:
    app = FastAPI()

    @app.get("/chat")
    async def chat() -> StreamingResponse:
        async def body() -> AsyncIterator[bytes]:
            with paid_call("chat", "main-model"):
                yield b"data: a\n\n"
            with paid_call("rerank", "reranker"):
                yield b"data: b\n\n"

        return StreamingResponse(body(), media_type="text/event-stream", headers={"X-Request-ID": "rid-1"})

    instrument_api(app)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://api") as client:  # type: ignore[arg-type]  # Starlette and HTTPX disagree on the ASGI send annotation.
        assert (await client.get("/chat")).status_code == 200

    server = by_name("GET /chat")
    assert attrs(server)["rag.paid_calls.total"] == 2
    assert attrs(server)["rag.request_id"] == "rid-1"


@pytest.mark.asyncio
async def test_probe_endpoints_suppress_server_and_client_spans() -> None:
    app = FastAPI()
    instrumented: list[bool] = []

    @app.get("/health")
    async def health() -> dict[str, str]:
        # Health probes call Qdrant/LLM through httpx; its instrumentation honours this flag.
        instrumented.append(is_instrumentation_enabled())
        return {"status": "ok"}

    instrument_api(app)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://api") as client:  # type: ignore[arg-type]  # Starlette and HTTPX disagree on the ASGI send annotation.
        assert (await client.get("/health")).status_code == 200

    assert instrumented == [False]
    assert finished() == ()


# ---------------------------------------------------------------------------
# API → worker
# ---------------------------------------------------------------------------


@pytest.fixture
def job_repository() -> Iterator[SQLAlchemyBackgroundJobRepository]:
    engine = create_engine("sqlite://")
    BackgroundJobModel.metadata.tables[BackgroundJobModel.__tablename__].create(engine)
    with Session(engine) as session:

        class AsyncSessionFacade:
            def add(self, instance: object) -> None:
                session.add(instance)

            async def execute(self, statement: Any) -> Any:
                return session.execute(statement)

            async def flush(self) -> None:
                session.flush()

            async def refresh(self, instance: object) -> None:
                session.refresh(instance)

            async def scalar(self, statement: Any) -> Any:
                return session.scalar(statement)

        yield SQLAlchemyBackgroundJobRepository(cast("AsyncSession", AsyncSessionFacade()))
    engine.dispose()


def uow_factory(repository: SQLAlchemyBackgroundJobRepository) -> object:
    class Factory:
        @asynccontextmanager
        async def create(self, *, master: bool = False) -> AsyncIterator[object]:
            del master
            yield SimpleNamespace(background_jobs=repository)

    return Factory()


async def run_inline(_uow: object, _job_id: int, action: Any, **_: object) -> None:
    await action()


@pytest.mark.asyncio
async def test_task_payload_contract_is_unchanged_by_tracing(
    job_repository: SQLAlchemyBackgroundJobRepository,
) -> None:
    payload: dict[str, object] = {"document_id": 3, "user_id": 1, "cost_principal": "user:1"}

    with step_span("DELETE /documents/3"):
        job = await job_repository.create(
            BackgroundJob(job_type=JobType.DOCUMENT_DELETION, dispatch_payload=dict(payload))
        )

    assert job.id is not None
    # Relays and handlers of the previous release validate exactly this payload.
    assert job.dispatch_payload == payload
    assert TASKS[JobType.DOCUMENT_DELETION].validate(payload, 1) == payload
    assert (await job_repository.trace_context(job.id) or {}).get("traceparent")


@pytest.mark.asyncio
async def test_untraced_submission_stores_no_context(
    job_repository: SQLAlchemyBackgroundJobRepository,
) -> None:
    job = await job_repository.create(BackgroundJob(job_type=JobType.SWEEP, dispatch_payload={"sweep_id": 1}))

    assert job.id is not None
    assert await job_repository.trace_context(job.id) is None


@pytest.mark.asyncio
async def test_worker_job_span_continues_the_api_trace(
    monkeypatch: pytest.MonkeyPatch, job_repository: SQLAlchemyBackgroundJobRepository
) -> None:
    monkeypatch.setattr(runtime, "run_tracked_job", run_inline)
    seen: dict[str, object] = {}

    @runtime.job(JobType.DOCUMENT_PROCESSING)
    async def process(_ctx: dict[str, Any], **kwargs: object) -> None:
        seen.update(kwargs)
        with paid_call("embeddings", "bge-m3"):
            pass

    with step_span("POST /documents") as api_span:
        job = await job_repository.create(
            BackgroundJob(job_type=JobType.DOCUMENT_PROCESSING, dispatch_payload={"document_id": 1})
        )

    assert job.id is not None
    await process({"uow_factory": uow_factory(job_repository), "job_try": 2}, job_id=job.id, document_id=1)

    job_span = by_name(f"job {JobType.DOCUMENT_PROCESSING}")
    assert job_span.kind == SpanKind.CONSUMER
    assert job_span.context.trace_id == api_span.get_span_context().trace_id
    assert job_span.parent is not None
    assert job_span.parent.span_id == api_span.get_span_context().span_id
    assert attrs(job_span)["rag.job.id"] == job.id
    assert attrs(job_span)["rag.paid_calls.embeddings"] == 1
    assert seen == {"job_id": job.id, "document_id": 1}


@pytest.mark.asyncio
async def test_trace_lookup_failure_does_not_block_the_job(monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenFactory:
        @asynccontextmanager
        async def create(self, *, master: bool = False) -> AsyncIterator[object]:
            del master
            message = "database unavailable"
            raise ConnectionError(message)
            yield  # type: ignore[unreachable]  # pragma: no cover  # makes create() an async generator

    monkeypatch.setattr(runtime, "run_tracked_job", run_inline)
    ran: list[int] = []

    @runtime.job(JobType.SWEEP)
    async def sweep(_ctx: dict[str, Any], **kwargs: Any) -> None:
        ran.append(kwargs["job_id"])

    await sweep({"uow_factory": BrokenFactory()}, job_id=5, sweep_id=1)

    assert ran == [5]
    assert by_name(f"job {JobType.SWEEP}").parent is None
