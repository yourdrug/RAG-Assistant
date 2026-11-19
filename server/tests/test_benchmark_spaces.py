"""Online runs bind retrieval, evidence, full evaluation and checkpoints to one space."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from qdrant_client.models import FieldCondition, MatchValue
from sqlalchemy.exc import IntegrityError
from starlette.requests import Request

from application.benchmark.dto import SweepCreateDTO
from application.benchmark.services import BenchmarkSweepService
from application.benchmark.sweep import SweepRun
from application.ports.route_configs import BenchmarkConfig
from application.spaces import KnowledgeSpaceService
from application.spaces.resolution import default_space_ref
from composition.cli.commands import benchmark as benchmark_cli
from composition.service_providers import build_sweep_engine
from domain.benchmark.question import BenchmarkQuestion
from domain.benchmark.run import BenchmarkRun
from domain.benchmark.sweep import BenchmarkSweep
from domain.knowledge.document import Document
from domain.knowledge.enums import DocumentStatus, DocumentVisibility
from domain.knowledge.space import KnowledgeSpace, SpaceRef
from domain.shared.exceptions import BusinessRuleViolationError, EntityNotFoundError, ValidationError
from infrastructure.benchmark.evaluation import runner
from infrastructure.benchmark.space_scope import active_sweep_space, space_corpus_metadata
from infrastructure.benchmark.sweep import data as sweep_data
from infrastructure.ml.evidence_profiles.profiles import GenericEvidenceProfile, current_evidence_profile
from infrastructure.repositories.benchmark.sqlalchemy_benchmark_run_repository import (
    SQLAlchemyBenchmarkRunRepository,
)
from infrastructure.repositories.benchmark.sqlalchemy_benchmark_sweep_repository import (
    SQLAlchemyBenchmarkSweepRepository,
)
from infrastructure.repositories.space.sqlalchemy_space_repository import SQLAlchemyKnowledgeSpaceRepository
from infrastructure.worker.registry import BenchmarkPayload
from presentation.api.dependencies import IdempotencyContext, JobSubmission
from presentation.api.helpers import sweep_create_to_dto, sweep_to_response
from presentation.api.routes.benchmark import run_benchmark
from presentation.api.schemas.benchmark import BenchmarkRequest, SweepCreateRequest
from tests.fakes import FakePackCatalog, FakeUnitOfWorkFactory


@pytest.fixture
def factory() -> FakeUnitOfWorkFactory:
    factory = FakeUnitOfWorkFactory()
    factory._uow.fake_knowledge_spaces._spaces[2] = KnowledgeSpace(id=2, name="Support", pack_key="support")
    factory._uow.fake_knowledge_spaces._spaces[3] = KnowledgeSpace(id=3, name="Internal", pack_key="internal")
    return factory


@pytest.mark.asyncio
async def test_explicit_space_is_required_in_multi_space_installation(factory: FakeUnitOfWorkFactory) -> None:
    assert await default_space_ref(factory, 2) == SpaceRef(2, "support")
    with pytest.raises(ValidationError):
        await default_space_ref(factory)
    with pytest.raises(EntityNotFoundError):
        await default_space_ref(factory, 99)


@pytest.mark.asyncio
async def test_sweep_selection_survives_api_dto_save_and_response(factory: FakeUnitOfWorkFactory) -> None:
    body = SweepCreateRequest(space_id=2, search_space={"top_k": {"values": [4]}})
    with pytest.raises(ValidationError):
        await BenchmarkSweepService(factory).create(SweepCreateDTO())
    saved = await BenchmarkSweepService(factory).create(sweep_create_to_dto(body))
    assert saved.space_id == 2
    assert sweep_to_response(saved).space_id == 2


@pytest.mark.asyncio
async def test_sweep_freezes_the_implicit_single_space() -> None:
    saved = await BenchmarkSweepService(FakeUnitOfWorkFactory()).create(SweepCreateDTO())
    assert saved.space_id == 1


def test_benchmark_selection_survives_durable_payload() -> None:
    request = BenchmarkRequest(space_id=3)
    payload = BenchmarkPayload(
        dataset=request.dataset, space_id=request.space_id, top_k=4, judge_model="judge"
    )
    assert BenchmarkPayload.model_validate(payload.model_dump()).space_id == 3


@pytest.mark.asyncio
async def test_parallel_sweeps_keep_space_and_profile_scoped(factory: FakeUnitOfWorkFactory) -> None:
    profiles = {key: GenericEvidenceProfile() for key in ("generic", "support", "internal")}
    engine = build_sweep_engine(factory, evidence_profiles=profiles)

    async def observe(space_id: int, pack_key: str) -> dict[str, Any]:
        async with engine._corpus(space_id) as metadata:
            await asyncio.sleep(0)
            assert active_sweep_space.get() == SpaceRef(space_id, pack_key)
            assert current_evidence_profile() is profiles[pack_key]
            return metadata

    support, internal = await asyncio.gather(observe(2, "support"), observe(3, "internal"))
    assert support["space_id"] == 2
    assert internal["space_id"] == 3
    assert support["corpus_namespace"] != internal["corpus_namespace"]
    assert active_sweep_space.get() is None


@pytest.mark.asyncio
async def test_phase_a_filters_both_searches_by_selected_space(
    factory: FakeUnitOfWorkFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    cached = AsyncMock(return_value=({}, {}, {}))
    monkeypatch.setattr(sweep_data, "cache_candidates", cached)
    engine = build_sweep_engine(factory, evidence_profiles={"support": GenericEvidenceProfile()})
    async with engine._corpus(2):
        await engine.cache_candidates([{"question": "Q"}], 10)
    search_filter = cached.call_args.kwargs["access_filter"]
    assert any(
        isinstance(condition, FieldCondition)
        and condition.key == "metadata.space_id"
        and isinstance(condition.match, MatchValue)
        and condition.match.value == 2
        for condition in search_filter.must
    )
    assert search_filter.must_not or search_filter.should or len(search_filter.must) > 1


@pytest.mark.asyncio
async def test_phase_b_passes_selected_space_to_full_benchmark(factory: FakeUnitOfWorkFactory) -> None:
    benchmark = SimpleNamespace(run=AsyncMock(return_value={"status": "done"}))
    engine = build_sweep_engine(factory, benchmark_service=benchmark)
    await engine.run_full_benchmark([], "judge", run=SweepRun(corpus={"space_id": 3}))
    assert benchmark.run.call_args.kwargs["space_id"] == 3


@pytest.mark.asyncio
async def test_full_runner_resolves_requested_space_before_evaluation(
    factory: FakeUnitOfWorkFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def execute(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        async with kwargs["scope"]() as execution:
            assert execution.space == SpaceRef(2, "support")
            assert execution.corpus_metadata["space_id"] == 2
        return []

    monkeypatch.setattr(runner, "execute_benchmark", execute)
    monkeypatch.setattr(runner, "build_run_manifest", lambda *_args: {})

    async def resolve(space_id: int | None) -> SpaceRef:
        return await default_space_ref(factory, space_id)

    await runner.run_benchmark_async(
        [{"question": "Q"}],
        4,
        "judge",
        space_id=2,
        space_resolver=resolve,
        rag_service=AsyncMock(),
    )


def test_corpus_identity_includes_space_even_in_one_pack() -> None:
    corpus = {"corpus_namespace": "same-collection"}
    first = space_corpus_metadata(corpus, SpaceRef(2, "support"))
    second = space_corpus_metadata(corpus, SpaceRef(3, "support"))
    assert first["corpus_namespace"] != second["corpus_namespace"]


@pytest.mark.asyncio
@pytest.mark.parametrize("space_id", [2, 99])
async def test_api_validates_and_persists_selected_space_before_enqueue(
    factory: FakeUnitOfWorkFactory, space_id: int
) -> None:
    jobs = AsyncMock()
    jobs.create_job.return_value = 8
    enqueuer = AsyncMock()
    user: Any = SimpleNamespace(kind="internal", role="admin", id=1)
    args: dict[str, Any] = {
        "request": Request({"type": "http", "method": "POST", "path": "/benchmark", "headers": []}),
        "req": BenchmarkRequest(space_id=space_id),
        "admin": user,
        "spaces": KnowledgeSpaceService(factory, FakePackCatalog()),
        "submission": JobSubmission(jobs=jobs, enqueuer=enqueuer, log=MagicMock()),
        "bench_cfg": BenchmarkConfig(
            data_dir="unused-benchmark-output", retriever_top_k=4, llm_model="judge"
        ),
        "idempotency": IdempotencyContext(key=None, store=MagicMock()),
    }

    if space_id == 99:
        with pytest.raises(EntityNotFoundError):
            await run_benchmark(**args)
        jobs.create_job.assert_not_awaited()
        enqueuer.enqueue_benchmark.assert_not_awaited()
    else:
        await run_benchmark(**args)
        assert jobs.create_job.call_args.kwargs["dispatch_payload"]["space_id"] == space_id
        assert enqueuer.enqueue_benchmark.call_args.kwargs["space_id"] == space_id


@pytest.mark.asyncio
async def test_regression_baseline_query_requires_the_same_space_and_pack() -> None:
    session = AsyncMock()
    session.scalar.return_value = None
    current = BenchmarkRun(id=5, dataset="same-dataset", config_json={"space_id": 2, "pack_key": "support"})
    await SQLAlchemyBenchmarkRunRepository(session).get_baseline(current)
    stmt = session.scalar.call_args.args[0].compile()
    assert {"space_id", "pack_key", "support", "2"} <= set(stmt.params.values())
    assert "judge_model" not in set(stmt.params.values())


@pytest.mark.asyncio
async def test_judged_baseline_query_requires_the_same_judge() -> None:
    session = AsyncMock()
    session.scalar.return_value = None
    current = BenchmarkRun(id=5, dataset="d", llm_evaluated=True, config_json={"judge_model": "judge-a"})
    await SQLAlchemyBenchmarkRunRepository(session).get_baseline(current)
    stmt = session.scalar.call_args.args[0].compile()
    assert {"judge_model", "judge-a"} <= set(stmt.params.values())


@pytest.mark.asyncio
@pytest.mark.parametrize("constraint", ["fk_benchmark_sweeps_space_id", "unrelated_constraint"])
async def test_space_deletion_preserves_sweep_binding_and_does_not_hide_other_errors(constraint: str) -> None:
    original: Any = RuntimeError("foreign key violation")
    original.diag = SimpleNamespace(constraint_name=constraint)
    failure = IntegrityError("delete", {}, original)
    session = AsyncMock()
    session.execute.side_effect = failure
    expected = BusinessRuleViolationError if constraint == "fk_benchmark_sweeps_space_id" else IntegrityError
    with pytest.raises(expected):
        await SQLAlchemyKnowledgeSpaceRepository(session).delete(2)


@pytest.mark.asyncio
async def test_cli_pins_space_in_job_and_broker_payload(
    factory: FakeUnitOfWorkFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    enqueue = AsyncMock()
    monkeypatch.setattr(benchmark_cli, "enqueue_benchmark", enqueue)
    monkeypatch.setattr(benchmark_cli, "close_arq_pool", AsyncMock())
    infra: Any = SimpleNamespace(db=SimpleNamespace(uow=factory))
    await factory._uow.benchmark_questions.create(BenchmarkQuestion(question="Q?", dataset="main"))
    await factory._uow.fake_documents.save(
        Document(
            filename="a",
            visibility=DocumentVisibility.INTERNAL_PUBLIC,
            space_id=2,
            status=DocumentStatus.DONE,
        )
    )
    jobs = await benchmark_cli.enqueue_runs(infra, ["main"], 7, "judge", space_id=2)
    saved = factory._uow.fake_background_jobs.jobs[jobs["main"]]
    assert saved.dispatch_payload is not None
    assert saved.dispatch_payload["space_id"] == 2
    assert enqueue.call_args.kwargs["space_id"] == 2
    # The worker compares this hash with the questions it loads.
    assert saved.dispatch_payload["dataset_hash"] == enqueue.call_args.kwargs["dataset_hash"]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [DocumentStatus.INDEXING, DocumentStatus.FAILED, None])
async def test_cli_refuses_a_run_before_the_corpus_is_searchable(
    factory: FakeUnitOfWorkFactory, monkeypatch: pytest.MonkeyPatch, status: DocumentStatus | None
) -> None:
    enqueue = AsyncMock()
    monkeypatch.setattr(benchmark_cli, "enqueue_benchmark", enqueue)
    monkeypatch.setattr(benchmark_cli, "close_arq_pool", AsyncMock())
    infra: Any = SimpleNamespace(db=SimpleNamespace(uow=factory))
    await factory._uow.benchmark_questions.create(BenchmarkQuestion(question="Q?", dataset="main"))
    await factory._uow.fake_documents.save(
        Document(
            filename="a",
            visibility=DocumentVisibility.INTERNAL_PUBLIC,
            space_id=2,
            status=DocumentStatus.DONE,
        )
    )

    if status is not None:
        await factory._uow.fake_documents.save(
            Document(filename="b", visibility=DocumentVisibility.INTERNAL_PUBLIC, space_id=2, status=status)
        )
    else:
        factory._uow.fake_documents._documents.clear()

    with pytest.raises(ValueError, match="not searchable"):
        await benchmark_cli.enqueue_runs(infra, ["main"], 7, "judge", space_id=2)

    enqueue.assert_not_awaited()
    assert not factory._uow.fake_background_jobs.jobs


@pytest.mark.asyncio
async def test_space_deleted_during_sweep_creation_is_reported_as_not_found() -> None:
    original: Any = RuntimeError("foreign key violation")
    original.diag = SimpleNamespace(constraint_name="fk_benchmark_sweeps_space_id")
    session = MagicMock()
    session.flush = AsyncMock(side_effect=IntegrityError("insert", {}, original))
    session.refresh = AsyncMock()
    with pytest.raises(EntityNotFoundError):
        await SQLAlchemyBenchmarkSweepRepository(session).create(BenchmarkSweep(space_id=2))
    session.refresh.assert_not_awaited()
