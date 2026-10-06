"""Failures discovered while reviewing the benchmark execution paths."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from config import get_setting
from domain.entities.benchmark_sweep import BenchmarkSweep
from domain.value_objects.benchmark_strategy import BenchmarkStrategy
from infrastructure.benchmark.runner import run_benchmark_async
from infrastructure.benchmark.sweep_engine import SweepEngine, SweepCancelled


@pytest.fixture
def optuna_engine(monkeypatch):
    engine = SweepEngine(uow_factory=None)
    monkeypatch.setattr(
        engine, "load_questions", AsyncMock(return_value=[{"question": "q", "source_hint": "doc"}])
    )
    monkeypatch.setattr(engine, "cache_candidates", AsyncMock(return_value=({}, {}, {})))
    monkeypatch.setattr(engine, "score_config_cheap", lambda cfg, *a: {"composite_score": cfg["top_k"]})
    return engine


def optuna_sweep():
    return BenchmarkSweep(
        strategy=BenchmarkStrategy.OPTUNA_TPE.value,
        search_space={"top_k": {"values": [2, 5]}, "fetch_k": {"min": 8, "max": 16}, "_n_trials": 3},
        top_n_llm=0,
    )


@pytest.mark.asyncio
async def test_optuna_accepts_scalar_fetch_k_and_integer_range_without_step(optuna_engine):
    results = await optuna_engine.run_sweep(optuna_sweep())
    assert len(results) == 3
    assert all(8 <= r["config"]["fetch_k"] <= 16 for r in results)
    assert optuna_engine.cache_candidates.call_args.args[1] == 16


@pytest.mark.asyncio
async def test_legacy_persisted_strategy_produces_same_tpe_results(optuna_engine):
    sweep = optuna_sweep()
    canonical = await optuna_engine.run_sweep(sweep)
    sweep.strategy = BenchmarkStrategy.SUCCESSIVE_HALVING.value
    legacy = await optuna_engine.run_sweep(sweep)
    assert legacy == canonical


@pytest.mark.asyncio
async def test_optuna_rejects_explicit_zero_step(optuna_engine):
    sweep = optuna_sweep()
    sweep.search_space["fetch_k"]["step"] = 0
    with pytest.raises(ValueError):
        await optuna_engine.run_sweep(sweep)


@pytest.mark.asyncio
async def test_optuna_checks_cancellation_between_trials_on_event_loop(optuna_engine):
    progress = []

    def callback(*args):
        asyncio.get_running_loop()
        progress.append(args)

    async def cancelled():
        return len(progress) == 1

    with pytest.raises(SweepCancelled):
        await optuna_engine.run_sweep(optuna_sweep(), progress_callback=callback, should_cancel=cancelled)
    assert len(progress) == 1


@pytest.mark.asyncio
async def test_optuna_keeps_event_loop_responsive_during_scoring(optuna_engine, monkeypatch):
    started = threading.Event()
    released = threading.Event()

    def score(cfg, *args):
        started.set()
        assert released.wait(2), "event loop was blocked during trial scoring"
        return {"composite_score": cfg["top_k"]}

    async def heartbeat():
        while not started.is_set():
            await asyncio.sleep(0)
        released.set()

    monkeypatch.setattr(optuna_engine, "score_config_cheap", score)
    task = asyncio.create_task(heartbeat())
    try:
        await optuna_engine.run_sweep(optuna_sweep())
        await task
    finally:
        released.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_full_pipeline_uses_requested_retrieval_limits(monkeypatch):
    questions = [{"question": "q"}]

    async def invoke(**kwargs):
        assert get_setting("retriever_top_k") == 7
        assert get_setting("retriever_fetch_k") == 23
        raise RuntimeError("limits checked")

    with pytest.raises(RuntimeError, match="limits checked"):
        await run_benchmark_async(
            questions, "unused", 7, "judge", fetch_k=23, rag_service=SimpleNamespace(invoke=invoke)
        )


@pytest.mark.asyncio
async def test_rag_generator_preserves_sweep_scope_and_restores_it_on_cancellation():
    from config import _settings_overrides
    from infrastructure.benchmark.answer_generators import RagBenchmarkGenerator

    inherited = {"retriever_fetch_k": 23, "dense_weight": 0.75, "cache_enabled": True}
    token = _settings_overrides.set(inherited)

    async def invoke(**kwargs):
        assert get_setting("retriever_fetch_k") == 23
        assert get_setting("dense_weight") == 0.75
        assert get_setting("cache_enabled") is False
        raise asyncio.CancelledError

    try:
        generator = RagBenchmarkGenerator(SimpleNamespace(invoke=invoke), 7, None)
        with pytest.raises(asyncio.CancelledError):
            await generator.generate({"question": "q"}, SimpleNamespace())
        assert _settings_overrides.get() == inherited
    finally:
        _settings_overrides.reset(token)


@pytest.mark.asyncio
async def test_async_judge_uses_explicit_model(monkeypatch):
    from infrastructure.benchmark import judge

    create = MagicMock(return_value=SimpleNamespace(score=8.0, reason="ok"))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(judge, "_get_judge_client", lambda model: client)
    monkeypatch.setattr(judge, "_get_judge_model", lambda: "default-model")
    result = await judge.judge_answer_async("question", "answer", "context", judge_model="requested-model")
    assert result["faithfulness"] == 8.0
    assert all(call.kwargs["model"] == "requested-model" for call in create.call_args_list)


def test_context_judge_uses_same_explicit_model(monkeypatch):
    from infrastructure.benchmark import judge, metrics

    score = MagicMock(return_value=SimpleNamespace(score=8.0, reason="ok"))
    monkeypatch.setattr(judge, "_get_judge_client", lambda model: object())
    monkeypatch.setattr(judge, "_get_judge_model", lambda: "default-model")
    monkeypatch.setattr(judge, "_judge_with_structured_output", score)
    result = metrics.compute_context_precision_recall(
        "question", "answer", context_override="context", judge_model="requested-model"
    )
    assert result["context_precision"] == 8.0
    assert all(call.args[2] == "requested-model" for call in score.call_args_list)


@pytest.mark.asyncio
async def test_version_update_rolls_back_rows_and_skips_cache_when_outbox_fails(monkeypatch):
    from datetime import date
    from application.services.act_versioning_service import ActVersioningService
    from domain.entities.act_version import ActVersion
    from fakes import FakeUnitOfWorkFactory

    factory = FakeUnitOfWorkFactory()
    cache = SimpleNamespace(invalidate_by_document_ids=AsyncMock())
    service = ActVersioningService(factory, SimpleNamespace(), cache)
    version = await factory._uow.act_versions.create(
        ActVersion(
            id=None,
            act_id=None,
            document_id=1,
            effective_from=date(2024, 1, 1),
            date_source="extracted",
        )
    )
    factory._uow.chunks._chunks.append(
        {
            "id": 1,
            "document_id": 1,
            "act_version_id": version.id,
            "is_current": True,
            "effective_from": date(2024, 1, 1),
        }
    )
    monkeypatch.setattr(
        factory._uow.vector_outbox, "enqueue", AsyncMock(side_effect=RuntimeError("outbox failed"))
    )
    with pytest.raises(RuntimeError, match="outbox failed"):
        await service.update_version(version.id, effective_from=date(2025, 1, 1))
    restored = await factory._uow.act_versions.get_by_id(version.id)
    assert restored.effective_from == date(2024, 1, 1)
    assert restored.date_source == "extracted"
    assert factory._uow.chunks._chunks[0]["effective_from"] == date(2024, 1, 1)
    assert factory._uow._rolled_back
    cache.invalidate_by_document_ids.assert_not_awaited()
