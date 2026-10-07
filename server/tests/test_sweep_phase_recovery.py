"""Resume after a judge failure without repeating retrieval search."""

from unittest.mock import AsyncMock

import pytest

from domain.entities.benchmark_sweep import BenchmarkSweep
from domain.value_objects.benchmark_strategy import BenchmarkStrategy
from infrastructure.benchmark.sweep_engine import SweepEngine
from infrastructure.benchmark.sweep_settings import LiveSweepSettings


@pytest.mark.asyncio
async def test_resume_restores_phase_a_and_invalidates_changed_search(tmp_path, monkeypatch):
    monkeypatch.setattr(LiveSweepSettings, "results_path", property(lambda self: str(tmp_path)))
    sweep = BenchmarkSweep(
        id=123,
        strategy=BenchmarkStrategy.RANDOM.value,
        search_space={"top_k": {"values": [1, 2, 3]}, "_n_random": 3},
        objective_weights={"hit_rate": 1},
        top_n_llm=1,
    )
    question = {"question": "q", "source_hint": "law.pdf"}
    first = SweepEngine(None)
    first.load_questions = AsyncMock(return_value=[question])
    first.cache_candidates = AsyncMock(return_value=({}, {}, {}))
    first._data.cache_reranker_scores = AsyncMock(return_value={})
    first.score_config_cheap = lambda *args: {"composite_score": 1}
    first.run_phase_b = AsyncMock(side_effect=RuntimeError("judge truncated"))
    with pytest.raises(RuntimeError, match="judge truncated"):
        await first.run_sweep(sweep, "judge")
    original_results = first.run_phase_b.call_args.args[0]

    # A fresh process restores the persisted shortlist, including random samples.
    resumed = SweepEngine(None)
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
