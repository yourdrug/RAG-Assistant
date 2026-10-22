"""Resume after a judge failure without repeating retrieval search."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from composition.service_providers import build_sweep_engine
from config import settings
from domain.benchmark.sweep import BenchmarkStrategy, BenchmarkSweep
from fakes import FakeUnitOfWorkFactory
from infrastructure.benchmark.checkpoint import DatabaseBenchmarkCheckpoints
from infrastructure.benchmark.sweep.settings import settings_fingerprint, settings_snapshot


@pytest.fixture
def checkpoints():
    factory = FakeUnitOfWorkFactory()

    def create(sweep_id):
        return DatabaseBenchmarkCheckpoints(factory, sweep_id)

    return create


@pytest.mark.asyncio
async def test_resume_restores_phase_a_and_invalidates_changed_search(checkpoints):
    sweep = BenchmarkSweep(
        id=123,
        strategy=BenchmarkStrategy.RANDOM.value,
        search_space={"top_k": {"values": [1, 2, 3]}, "_n_random": 3},
        objective_weights={"hit_rate": 1},
        top_n_llm=1,
    )
    question = {"question": "q", "source_hint": "law.pdf"}
    first = build_sweep_engine(None, checkpoint_factory=checkpoints)
    first.load_questions = AsyncMock(return_value=[question])
    first.cache_candidates = AsyncMock(return_value=({}, {}, {}))
    first._data.cache_reranker_scores = AsyncMock(return_value={})
    first.score_config_cheap = lambda *args: {"composite_score": 1}
    first.run_phase_b = AsyncMock(side_effect=RuntimeError("judge truncated"))
    with pytest.raises(RuntimeError, match="judge truncated"):
        await first.run_sweep(sweep, "judge")
    original_results = first.run_phase_b.call_args.args[0]

    # A fresh process restores the persisted shortlist, including random samples.
    resumed = build_sweep_engine(None, checkpoint_factory=checkpoints)
    resumed.load_questions = AsyncMock(side_effect=AssertionError("dataset loaded again"))
    resumed.cache_candidates = AsyncMock(side_effect=AssertionError("retrieval repeated"))
    resumed._data.cache_reranker_scores = AsyncMock(side_effect=AssertionError("reranking repeated"))
    resumed.run_phase_b = AsyncMock(return_value=[])
    progress = AsyncMock()
    await resumed.run_sweep(sweep, "judge", progress)
    assert resumed.run_phase_b.call_args.args[0] == original_results
    resumed.cache_candidates.assert_not_awaited()
    resumed._data.cache_reranker_scores.assert_not_awaited()
    assert progress.call_args.args[2]["resumed"] is True

    sweep.search_space["top_k"]["values"] = [4]
    resumed.cache_candidates = AsyncMock(return_value=({}, {}, {}))
    resumed._data.cache_reranker_scores = AsyncMock(return_value={})
    resumed.score_config_cheap = lambda *args: {"composite_score": 1}
    await resumed.run_sweep(sweep, "judge")
    resumed.cache_candidates.assert_awaited_once()
    assert all(r["config"] == {"top_k": 4} for r in resumed.run_phase_b.call_args.args[0])


@pytest.mark.asyncio
async def test_concurrent_sweeps_keep_separate_run_state(checkpoints, monkeypatch):
    reached = set()
    ready = asyncio.Event()
    observed = []

    async def phase_a(sweep, *args):
        reached.add(sweep.id)

        if len(reached) == 2:
            ready.set()

        await ready.wait()
        return [{"config": {"top_k": sweep.id}, "composite_score": 1}]

    async def full(**kwargs):
        observed.append((kwargs["checkpoints"]._sweep_id, kwargs["top_k"], kwargs["questions"][0]["id"]))
        return {"hit_rate": 1}

    engine = build_sweep_engine(
        None, benchmark_service=SimpleNamespace(run=full), checkpoint_factory=checkpoints
    )
    monkeypatch.setattr(engine, "evaluate_phase_a", phase_a)
    monkeypatch.setattr(
        engine,
        "load_questions",
        AsyncMock(
            side_effect=lambda dataset: [{"id": int(dataset), "question": dataset, "source_hint": "law.pdf"}]
        ),
    )
    await asyncio.gather(
        *(
            engine.run_sweep(
                BenchmarkSweep(
                    id=i,
                    dataset=str(i),
                    search_space={"top_k": {"values": [i]}},
                    objective_weights={"hit_rate": 1},
                    top_n_llm=1,
                ),
                "judge",
            )
            for i in (1, 2)
        )
    )
    assert sorted(observed) == [(1, 1, 1), (2, 2, 2)]


def test_settings_fingerprint_excludes_secrets_and_operational_judge_limits(monkeypatch):
    initial = settings_fingerprint()
    monkeypatch.setattr(settings, "openrouter_api_key", "private-key")
    monkeypatch.setattr(settings, "benchmark_judge_initial_tokens", 9999)
    monkeypatch.setattr(settings, "benchmark_judge_retry_tokens", 9999)
    monkeypatch.setattr(settings, "benchmark_judge_max_concurrent", 9)
    monkeypatch.setattr(
        settings, "benchmark_judge_grouped_enabled", not settings.benchmark_judge_grouped_enabled
    )
    assert settings_fingerprint() == initial
    assert all(
        not any(part in key for part in ("key", "password", "secret", "url", "dir"))
        for key in settings_snapshot()
    )
    monkeypatch.setattr(settings, "rrf_k", settings.rrf_k + 1)
    assert settings_fingerprint() != initial
