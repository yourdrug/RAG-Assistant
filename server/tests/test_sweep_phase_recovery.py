"""Resume after a judge failure without repeating retrieval search."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, Mock

from pydantic import SecretStr
import pytest

from composition.service_providers import build_sweep_engine
from config import settings
from domain.benchmark.sweep import BenchmarkStrategy, BenchmarkSweep
from infrastructure.benchmark.checkpoint import DatabaseBenchmarkCheckpoints
from infrastructure.benchmark.sweep.settings import settings_fingerprint, settings_snapshot
from tests.fakes import FakeUnitOfWorkFactory


if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture
def checkpoints() -> Callable[[int], DatabaseBenchmarkCheckpoints]:
    factory = FakeUnitOfWorkFactory()

    def create(sweep_id: int) -> DatabaseBenchmarkCheckpoints:
        return DatabaseBenchmarkCheckpoints(factory, sweep_id)

    return create


async def checkpoint_failed_judge(
    monkeypatch: pytest.MonkeyPatch,
    checkpoints: Callable[[int], DatabaseBenchmarkCheckpoints],
    sweep: BenchmarkSweep,
    question: dict[str, Any],
) -> list[dict[str, Any]]:
    first = build_sweep_engine(FakeUnitOfWorkFactory(), checkpoint_factory=checkpoints)
    first_load_questions = AsyncMock(return_value=[question])
    monkeypatch.setattr(first, "load_questions", first_load_questions)
    first_cache_candidates = AsyncMock(return_value=({}, {}, {}))
    monkeypatch.setattr(first, "cache_candidates", first_cache_candidates)
    first_cache_reranker_scores = AsyncMock(return_value={})
    monkeypatch.setattr(first._data, "cache_reranker_scores", first_cache_reranker_scores)
    first_score_config_cheap = Mock(return_value={"composite_score": 1})
    monkeypatch.setattr(first, "score_config_cheap", first_score_config_cheap)
    first_run_phase_b = AsyncMock(side_effect=RuntimeError("judge truncated"))
    monkeypatch.setattr(first, "run_phase_b", first_run_phase_b)
    with pytest.raises(RuntimeError, match="judge truncated"):
        await first.run_sweep(sweep, "judge")
    original_results: list[dict[str, Any]] = first_run_phase_b.call_args.args[0]
    return original_results


@pytest.mark.asyncio
async def test_resume_restores_phase_a_and_invalidates_changed_search(
    monkeypatch: pytest.MonkeyPatch, checkpoints: Callable[[int], DatabaseBenchmarkCheckpoints]
) -> None:
    sweep = BenchmarkSweep(
        id=123,
        strategy=BenchmarkStrategy.RANDOM,
        search_space={"top_k": {"values": [1, 2, 3]}, "_n_random": 3},
        objective_weights={"hit_rate": 1},
        top_n_llm=1,
    )
    question = {"question": "q", "source_hint": "law.pdf"}
    original_results = await checkpoint_failed_judge(monkeypatch, checkpoints, sweep, question)

    # A fresh process restores the persisted shortlist, including random samples.
    resumed = build_sweep_engine(FakeUnitOfWorkFactory(), checkpoint_factory=checkpoints)
    resumed_load_questions = AsyncMock(side_effect=AssertionError("dataset loaded again"))
    monkeypatch.setattr(resumed, "load_questions", resumed_load_questions)
    resumed_cache_candidates = AsyncMock(side_effect=AssertionError("retrieval repeated"))
    monkeypatch.setattr(resumed, "cache_candidates", resumed_cache_candidates)
    resumed_cache_reranker_scores = AsyncMock(side_effect=AssertionError("reranking repeated"))
    monkeypatch.setattr(resumed._data, "cache_reranker_scores", resumed_cache_reranker_scores)
    resumed_run_phase_b = AsyncMock(return_value=[])
    monkeypatch.setattr(resumed, "run_phase_b", resumed_run_phase_b)
    progress = AsyncMock()
    await resumed.run_sweep(sweep, "judge", progress)
    assert resumed_run_phase_b.call_args.args[0] == original_results
    resumed_cache_candidates.assert_not_awaited()
    resumed_cache_reranker_scores.assert_not_awaited()
    assert progress.call_args.args[2]["resumed"] is True

    sweep.search_space["top_k"]["values"] = [4]
    resumed_cache_candidates = AsyncMock(return_value=({}, {}, {}))
    monkeypatch.setattr(resumed, "cache_candidates", resumed_cache_candidates)
    resumed_cache_reranker_scores = AsyncMock(return_value={})
    monkeypatch.setattr(resumed._data, "cache_reranker_scores", resumed_cache_reranker_scores)
    resumed_score_config_cheap = Mock(return_value={"composite_score": 1})
    monkeypatch.setattr(resumed, "score_config_cheap", resumed_score_config_cheap)
    await resumed.run_sweep(sweep, "judge")
    resumed_cache_candidates.assert_awaited_once()
    assert all(r["config"] == {"top_k": 4} for r in resumed_run_phase_b.call_args.args[0])


@pytest.mark.asyncio
async def test_concurrent_sweeps_keep_separate_run_state(
    checkpoints: Callable[[int], DatabaseBenchmarkCheckpoints], monkeypatch: pytest.MonkeyPatch
) -> None:
    reached: set[int | None] = set()
    ready = asyncio.Event()
    observed: list[tuple[int, int, int]] = []

    async def phase_a(sweep: BenchmarkSweep, *args: object) -> list[dict[str, Any]]:
        reached.add(sweep.id)

        if len(reached) == 2:
            ready.set()

        await ready.wait()
        return [{"config": {"top_k": sweep.id}, "composite_score": 1}]

    async def full(**kwargs: Any) -> dict[str, Any]:
        observed.append((kwargs["checkpoints"]._sweep_id, kwargs["top_k"], kwargs["questions"][0]["id"]))
        return {"hit_rate": 1}

    engine = build_sweep_engine(
        FakeUnitOfWorkFactory(), benchmark_service=SimpleNamespace(run=full), checkpoint_factory=checkpoints
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


def test_settings_fingerprint_excludes_secrets_and_operational_judge_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = settings_fingerprint()
    monkeypatch.setattr(settings, "openrouter_api_key", SecretStr("private-key"))
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
