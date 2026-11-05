"""Regressions for queue scope, temporal candidates and admitted ML generations."""

from __future__ import annotations

import asyncio
from datetime import date
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

from fastapi import HTTPException
import pytest
from qdrant_client.local.payload_filters import check_filter

from application.ports.job_enqueuer import DuplicateJobError
from application.ports.ml_clients import MLClientPort
from application.rag.dto import PreparedQuery
from config import get_request_settings, settings
from domain.identity.roles import UserKind, UserRole
from domain.identity.user_context import UserContext
from domain.knowledge.enums import DocumentVisibility
from domain.rag.context import RagContext
from infrastructure.benchmark.sweep.data import SweepDataSource
from infrastructure.ml.cache.keys import CacheKeyBuilder
from infrastructure.ml.clients.client_registry import MLClientRegistry
from infrastructure.ml.clients.managed_llm import ManagedLLM
from infrastructure.ml.rag.rag_steps import RagOperations
from infrastructure.repositories.vector.acl import build_search_filter
from infrastructure.worker import queue
from presentation.api.dependencies import IdempotencyContext, JobSubmission
from presentation.api.routes import ingest as ingest_routes


if TYPE_CHECKING:
    from qdrant_client.models import Filter


@pytest.mark.asyncio
@pytest.mark.parametrize("single", [False, True])
@pytest.mark.parametrize(
    "difference",
    [
        {"client_id": 2},
        {"group_id": 3},
        {"visibility": DocumentVisibility.INTERNAL_PRIVATE.value},
        {"domain": "legal"},
    ],
)
async def test_ingestion_scope_has_distinct_queue_identity(
    monkeypatch: pytest.MonkeyPatch, single: bool, difference: dict[str, Any]
) -> None:
    enqueue = AsyncMock(return_value=object())
    monkeypatch.setattr(queue, "enqueue_arq", enqueue)
    arguments: dict[str, Any] = {"domain": "general", "job_id": 10, "client_id": 1}
    publish: Any

    if single:
        publish = queue.enqueue_ingest_file
        arguments["resolved"] = "docs/file.txt"
    else:
        publish = queue.enqueue_ingest
        arguments.update(resolved_dir="docs/", reset=False)

    await publish(**arguments)
    first_id = enqueue.call_args.kwargs["arq_job_id"]
    await publish(**{**arguments, "job_id": 11})
    assert enqueue.call_args.kwargs["arq_job_id"] == first_id
    await publish(**{**arguments, **difference, "job_id": 12})
    assert enqueue.call_args.kwargs["arq_job_id"] != first_id


@pytest.mark.asyncio
async def test_force_changes_identity_and_duplicate_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    enqueue = AsyncMock(return_value=object())
    monkeypatch.setattr(queue, "enqueue_arq", enqueue)
    arguments: dict[str, Any] = {"resolved": "docs/file.txt", "domain": "general", "job_id": 1}
    await queue.enqueue_ingest_file(**arguments)
    first_id = enqueue.call_args.kwargs["arq_job_id"]
    await queue.enqueue_ingest_file(**arguments, force=True)
    assert enqueue.call_args.kwargs["arq_job_id"] != first_id
    enqueue.return_value = None
    with pytest.raises(DuplicateJobError):
        await queue.enqueue_ingest_file(**arguments)
    with pytest.raises(DuplicateJobError):
        await queue.enqueue_ingest(resolved_dir="docs/", reset=False, domain="general", job_id=2)


@pytest.mark.asyncio
async def test_sweep_dense_and_sparse_share_production_temporal_scope() -> None:
    today = date.today()  # noqa: DTZ011 — the production temporal contract uses local dates.
    metadata = {
        "visibility": DocumentVisibility.INTERNAL_PUBLIC.value,
        "is_current": True,
    }
    points = [
        SimpleNamespace(id=1, score=1.0, payload={"page_content": "evidence", "metadata": metadata}),
        SimpleNamespace(
            id=2,
            score=1.0,
            payload={"page_content": "expired", "metadata": {**metadata, "effective_to": today.isoformat()}},
        ),
        SimpleNamespace(
            id=3,
            score=1.0,
            payload={"page_content": "future", "metadata": {**metadata, "effective_from": "2099-01-01"}},
        ),
        SimpleNamespace(
            id=4,
            score=1.0,
            payload={"page_content": "private", "metadata": {**metadata, "visibility": "client_private"}},
        ),
    ]
    filters = []

    def search(*, query_filter: Filter, **kwargs: Any) -> list[Any]:
        filters.append(query_filter)
        return [
            point
            for point in points
            if check_filter(query_filter.model_copy(deep=True), point.payload, point.id)
        ]

    clients = MagicMock()
    clients.qdrant_client.return_value = SimpleNamespace(search=search)
    clients.embeddings.return_value = SimpleNamespace(embed_query_sync=lambda _query: [0.1])
    dense, sparse, candidates = await SweepDataSource(MagicMock(), clients).cache_candidates(
        [{"question": "evidence"}], 10
    )
    assert set(candidates) == {"1"}
    assert [key for key, *_ in dense["evidence"]] == ["1"]
    assert [key for key, _ in sparse["evidence"]] == ["1"]
    production = build_search_filter({"id": 0, "kind": UserKind.INTERNAL, "role": UserRole.ADMIN}, [])
    assert filters == [production, production]


class ClientFamily(ManagedLLM):
    async def embed_query(self, text: str) -> list[float]:
        return self.embed_query_sync(text)

    def embed_query_sync(self, text: str) -> list[float]:
        return [float(len(text))]

    async def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [1.0] * len(pairs)


def install_family(registry: MLClientRegistry, client: ClientFamily) -> None:
    registry.install_clients(
        {
            "cached_embeddings": client,
            "cached_reranker": client,
            "cached_llm": client,
            "cached_fast_llm": client,
            "cached_instructor": client,
            "cached_llm_breadth_cache": {"narrow": client, "broad": client},
        }
    )


@pytest.mark.asyncio
async def test_request_keeps_settings_cache_model_and_client_family_during_reload(  # noqa: PLR0915 — exercise one full reload lifecycle.
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "llm_provider", "openrouter")
    monkeypatch.setattr(settings, "openrouter_model", "local-model")
    monkeypatch.setattr(settings, "openrouter_model", "remote-model")
    monkeypatch.setattr(settings, "rag_retrieval_score_threshold", 0.1)
    old_close, new_close = AsyncMock(), AsyncMock()
    old = ClientFamily(SimpleNamespace(close=old_close))
    new = ClientFamily(SimpleNamespace(close=new_close))
    registry = MLClientRegistry()
    registry.corpus_resources.qdrant_factory = MagicMock()
    install_family(registry, old)
    operations = RagOperations(registry, document_access=MagicMock())
    ctx = RagContext(user=UserContext(user_id=1, user_kind=UserKind.INTERNAL, user_role=UserRole.USER))
    try:
        async with operations.corpus():
            request = operations.request("Q", [], ctx)
            assert request.model == "remote-model"
            query = PreparedQuery(request, "Q")
            key = CacheKeyBuilder.build(query)

            async def reload() -> None:
                monkeypatch.setattr(settings, "openrouter_model", "new-model")
                monkeypatch.setattr(settings, "rag_retrieval_score_threshold", 0.9)
                install_family(registry, new)

            await asyncio.create_task(reload())
            assert get_request_settings().openrouter_model == "remote-model"
            assert get_request_settings().rag_retrieval_score_threshold == 0.1
            assert registry.embeddings() is old
            assert registry.fast_llm() is old
            assert registry.llm_for_breadth("narrow") is old
            assert registry.llm_for_breadth("broad") is old
            assert registry.instructor_client is old
            assert CacheKeyBuilder.build(query) == key
            old_close.assert_not_awaited()
        await old.close()
        old_close.assert_awaited_once()
        async with operations.corpus():
            updated = operations.request("Q", [], ctx)
            assert updated.model == "new-model"
            assert registry.llm_for_breadth("broad") is new
            assert CacheKeyBuilder.build(PreparedQuery(updated, "Q")) != key
            # A previously admitted request has an immutable cache identity even outside its lease.
            assert CacheKeyBuilder.build(PreparedQuery(request, "Q")) == key
    finally:
        await registry.close()


def test_ml_runtime_contract_has_no_local_bm25() -> None:
    for name in ("bm25_index", "bm25_update", "ensure_bm25_loaded", "invalidate_bm25"):
        assert not hasattr(MLClientPort, name)
        assert not hasattr(MLClientRegistry, name)


@pytest.mark.asyncio
@pytest.mark.parametrize("single", [False, True])
async def test_ingestion_duplicate_fails_created_job_and_returns_conflict(single: bool) -> None:
    request = MagicMock(method="POST")
    request.url.path = "/ingest/file" if single else "/ingest"
    jobs = MagicMock(create_job=AsyncMock(return_value=42), mark_failed=AsyncMock())
    enqueuer = MagicMock()
    enqueuer.enqueue_ingest = AsyncMock(side_effect=DuplicateJobError("already queued"))
    enqueuer.enqueue_ingest_file = AsyncMock(side_effect=DuplicateJobError("already queued"))
    service = MagicMock()
    service.resolve_docs_dir.return_value = "docs/"
    service.resolve_ingest_target.return_value = "docs/file.txt"
    parameters: dict[str, Any] = {
        "request": request,
        "admin": MagicMock(),
        "service": service,
        "submission": JobSubmission(jobs=jobs, enqueuer=enqueuer, log=MagicMock()),
        "idempotency": IdempotencyContext(key=None, store=MagicMock()),
        "options": ingest_routes.IngestOptions(),
    }
    action: Any = ingest_routes.ingest_documents

    if single:
        action = ingest_routes.ingest_single_file
        parameters["options"] = ingest_routes.IngestFileOptions(file_path="docs/file.txt")

    with pytest.raises(HTTPException) as error:
        await action(**parameters)
    assert error.value.status_code == 409
    jobs.mark_failed.assert_awaited_once_with(42, "already queued")
