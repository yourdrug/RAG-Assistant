"""Regressions for corpus switches during evaluations."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from functools import partial
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, Mock

import pytest
from qdrant_client.models import Filter

from application.benchmark import jobs as benchmark_jobs
from application.benchmark.jobs import BenchmarkJobService
from composition.service_providers import build_sweep_engine
from domain.benchmark.sweep import BenchmarkSweep
from domain.knowledge.access_control import VisibilityCondition
from domain.knowledge.enums import DocumentVisibility
from infrastructure.benchmark.checkpoint import DatabaseBenchmarkCheckpoints
from infrastructure.benchmark.corpus import benchmark_scope, corpus_metadata
from infrastructure.benchmark.evaluation import case_evaluator, runner
from infrastructure.benchmark.reporting import progress
from infrastructure.benchmark.sweep import candidates as sweep_candidates
from infrastructure.corpus.snapshot import BoundQdrantClient, CorpusSnapshot
from infrastructure.ml.clients.client_registry import MLClientRegistry
from infrastructure.worker import runtime, tasks
from tests.fakes import FakeUnitOfWorkFactory


if TYPE_CHECKING:
    from pathlib import Path
    from unittest.mock import MagicMock

    from application.benchmark.export_service import BenchmarkExportService
    from application.benchmark.result_service import BenchmarkResultService
    from application.uow.factory import UnitOfWorkFactory


def corpus_pair() -> tuple[MLClientRegistry, BoundQdrantClient, BoundQdrantClient]:
    registry = MLClientRegistry()
    snapshot = CorpusSnapshot("http://qdrant:6333", "docs", "v1", "identity", "generation1")
    old = BoundQdrantClient(Mock(), snapshot)
    new = BoundQdrantClient(Mock(), replace(snapshot, physical_collection="v2", generation="generation2"))
    registry.install_corpus(old)
    return registry, old, new


@pytest.mark.asyncio
async def test_benchmark_pins_all_parallel_and_queued_questions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    registry, old, new = corpus_pair()
    observed = []
    switched = asyncio.Event()

    async def evaluate(
        self: object, idx: Any, question: dict[str, Any], run_idx: Any, ctx: Any, **kwargs: Any
    ) -> Any:
        if idx == 1:
            registry.install_corpus(new)
            switched.set()

        await switched.wait()
        await asyncio.sleep(0)
        observed.append(registry.qdrant_client())
        assert corpus_metadata(registry.corpus_snapshot)["corpus_generation"] == "generation1"
        cast("MagicMock", old.client.close).assert_not_called()
        return {
            "id": idx,
            "question": question["question"],
            "generator_metrics": {},
            "retriever_metrics": {},
            "latency_sec": 1,
        }

    monkeypatch.setattr(case_evaluator.BenchmarkCaseEvaluator, "evaluate", evaluate)
    monkeypatch.setattr(progress, "log_question_result", lambda *_args: None)
    monkeypatch.setattr(progress, "log_summary", lambda *_args: None)
    try:
        await runner.run_benchmark_async(
            [{"question": "q", "annotations": {"expected_refusal": True}}] * 3,
            1,
            "judge",
            max_concurrent=2,
            rag_service=SimpleNamespace(),
            corpus_resource=registry,
        )
        assert observed == [old] * 3
        assert list(tmp_path.iterdir()) == []  # noqa: ASYNC240 - Small local fixture I/O in the test setup.
        await old.aclose()
        cast("MagicMock", old.client.close).assert_called_once()
        assert registry.qdrant_client() is new
    finally:
        await registry.close()


@pytest.mark.asyncio
async def test_candidate_cache_keeps_client_across_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    registry, old, new = corpus_pair()
    cast("MagicMock", old.client.search).return_value = []
    registry.cached_embeddings = Mock()
    registry.cached_embeddings.embed_query_sync.return_value = [1.0]

    def search(*args: Any, **kwargs: Any) -> list[Any]:
        registry.install_corpus(new)
        return []

    cast("MagicMock", old.client.search).side_effect = search
    try:
        await sweep_candidates.cache_candidates(
            [{"question": "alpha"}, {"question": "beta"}],
            1,
            ml_clients=registry,
            access_filter=Filter(),
            visibility_conditions=[VisibilityCondition(DocumentVisibility.INTERNAL_PUBLIC)],
            user_id=0,
            user_group_ids=[],
        )
        assert cast("MagicMock", old.client.search).call_count == 4
        cast("MagicMock", new.client.search).assert_not_called()
        assert all(
            call.kwargs["collection_name"] == "v1"
            for call in cast("MagicMock", old.client.search).call_args_list
        )
    finally:
        await old.aclose()
        await registry.close()


@pytest.mark.asyncio
async def test_sweep_pins_preparation_and_rejects_resume_on_changed_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, old, new = corpus_pair()
    factory = FakeUnitOfWorkFactory()

    def checkpoints(sweep_id: int) -> DatabaseBenchmarkCheckpoints:
        return DatabaseBenchmarkCheckpoints(factory, sweep_id)

    engine = build_sweep_engine(FakeUnitOfWorkFactory(), ml_clients=registry, checkpoint_factory=checkpoints)
    monkeypatch.setattr(
        engine, "load_questions", AsyncMock(return_value=[{"question": "q", "source_hint": "law.pdf"}])
    )

    async def phase_a(*args: Any) -> list[dict[str, Any]]:
        registry.install_corpus(new)
        return [{"config": {"top_k": 1}, "composite_score": 1}]

    async def phase_b(*args: Any, **kwargs: Any) -> list[Any]:
        assert registry.qdrant_client() is old
        assert kwargs["run"].corpus == corpus_metadata(old.snapshot)
        cast("MagicMock", old.client.close).assert_not_called()
        return []

    monkeypatch.setattr(engine, "evaluate_phase_a", phase_a)
    monkeypatch.setattr(engine, "run_phase_b", phase_b)
    sweep = BenchmarkSweep(id=123, search_space={"top_k": {"values": [1]}}, objective_weights={"hit_rate": 1})
    try:
        await engine.run_sweep(sweep, "judge")
        with pytest.raises(ValueError, match="corpus changed"):
            await engine.run_sweep(sweep, "judge")
    finally:
        await old.aclose()
        await registry.close()


@pytest.mark.asyncio
async def test_worker_report_uses_pinned_corpus_after_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    registry, old, new = corpus_pair()
    service = SimpleNamespace(get_job_result=AsyncMock(return_value=None), record_result=AsyncMock())
    service.record_result.return_value = SimpleNamespace(id=None)
    container = SimpleNamespace(
        infrastructure=SimpleNamespace(
            db=SimpleNamespace(uow_factory=object()), ml=SimpleNamespace(ml_clients=registry)
        ),
        application=SimpleNamespace(rag_service=SimpleNamespace()),
    )
    monkeypatch.setattr(
        benchmark_jobs, "load_benchmark_questions", AsyncMock(return_value=[{"question": "q"}])
    )

    async def evaluate(**kwargs: Any) -> dict[str, Any]:
        registry.install_corpus(new)
        return {"results": [{}], "total_time_sec": 1}

    async def tracked(factory: object, job_id: object, action: Any, **kwargs: Any) -> None:
        await action()

    container.infrastructure.ml = SimpleNamespace(
        ml_clients=registry, benchmark_service=SimpleNamespace(run=evaluate)
    )
    monkeypatch.setattr(runtime, "run_tracked_job", tracked)
    try:
        await tasks.run_benchmark(
            {
                "uow_factory": object(),
                "benchmark_job": BenchmarkJobService(
                    cast("UnitOfWorkFactory", object()),
                    container.infrastructure.ml.benchmark_service,
                    cast("BenchmarkResultService", service),
                    cast("BenchmarkExportService", object()),
                    partial(benchmark_scope, registry),
                ),
            },
            dataset="main",
            top_k=1,
            judge_model="judge",
            job_id=1,
        )
        report = service.record_result.call_args.args[0]
        assert report.config_json["collection_name"] == "v1"
        assert report.config_json["corpus_generation"] == "generation1"
        assert report.config_json["embedding_identity"] == "identity"
    finally:
        await old.aclose()
        await registry.close()
