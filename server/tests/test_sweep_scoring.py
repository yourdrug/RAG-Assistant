"""Characterization and regression contracts for sweep scoring and final selection."""

from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from config import get_setting
from domain.value_objects.benchmark_scoring import DEFAULT_OBJECTIVE_WEIGHTS
from infrastructure.benchmark.sweep_full_evaluation import SweepFullEvaluator
from infrastructure.benchmark.sweep_scoring import compute_composite_score
from infrastructure.benchmark.sweep_settings import LiveSweepSettings
from presentation.api.schemas.benchmark import SweepCreateRequest


def test_retrieval_only_score_uses_available_weights():
    assert compute_composite_score(
        {"hit_rate": 1, "mrr": 0.5}, {"hit_rate": 0.4, "mrr": 0.2}
    ) == pytest.approx(5 / 6)


@pytest.mark.asyncio
async def test_full_evaluation_runs_shortlist_and_restores_settings():
    previous_top_k = get_setting("retriever_top_k")
    results = [
        {"config": {"top_k": k}, "avg_hit_rate": 1, "avg_mrr": 1, "composite_score": 1} for k in (5, 2)
    ]
    run = AsyncMock(return_value={"hit_rate": 1, "avg_mrr": 1, "avg_faithfulness": 8, "avg_relevancy": 7})
    evaluated = await SweepFullEvaluator(LiveSweepSettings()).evaluate(
        results,
        2,
        "judge",
        [{"question": "q"}],
        {"hit_rate": 0.4, "faithfulness": 0.3, "relevancy": 0.3},
        None,
        run,
    )
    assert run.await_count == 2
    assert all(r["llm_evaluated"] for r in evaluated)
    assert [r["config"]["top_k"] for r in evaluated] == [5, 2]
    assert get_setting("retriever_top_k") == previous_top_k


def test_normalization_makes_weights_proportional_to_influence():
    assert compute_composite_score({"hit_rate": 1, "faithfulness": 0, "relevancy": 0}) == pytest.approx(0.4)
    assert compute_composite_score({"hit_rate": 0, "faithfulness": 10, "relevancy": 0}) == pytest.approx(0.3)
    assert compute_composite_score({"hit_rate": 1, "faithfulness": 10, "relevancy": 10}) == pytest.approx(1)


def test_mrr_and_correctness_have_explicit_normalized_influence():
    assert compute_composite_score(
        {"mrr": 0.5, "correctness": 10}, {"mrr": 0.2, "correctness": 0.8}
    ) == pytest.approx(0.9)


def test_explicit_weights_do_not_add_hidden_metrics():
    metrics = {"hit_rate": 1, "mrr": 0, "faithfulness": 10, "relevancy": 10, "correctness": 0}
    assert compute_composite_score(
        metrics, {"hit_rate": 0.4, "faithfulness": 0.3, "relevancy": 0.3}
    ) == pytest.approx(1)
    request = SweepCreateRequest(search_space={}, objective_weights={"hit_rate": 1})
    assert request.objective_weights == {**dict.fromkeys(DEFAULT_OBJECTIVE_WEIGHTS, 0), "hit_rate": 1}
    assert SweepCreateRequest(search_space={}).objective_weights == DEFAULT_OBJECTIVE_WEIGHTS


@pytest.mark.parametrize(
    "weights",
    [
        {},
        {"hit_rate": -1},
        {"hit_rate": float("nan")},
        {"hit_rate": float("inf")},
        {"typo": 1},
        {"faithfulness": 1},
    ],
)
def test_invalid_objectives_rejected(weights):
    with pytest.raises(ValidationError):
        SweepCreateRequest(search_space={}, objective_weights=weights)


@pytest.mark.asyncio
async def test_unchecked_config_cannot_beat_full_finalists():
    results = [
        {"config": {"top_k": k}, "composite_score": score} for k, score in [(5, 1), (3, 0.9), (2, 0.8)]
    ]
    run = AsyncMock(
        side_effect=[
            {"hit_rate": 0, "avg_faithfulness": 1, "avg_relevancy": 1},
            {"hit_rate": 0, "avg_faithfulness": 2, "avg_relevancy": 2},
        ]
    )
    ranked = await SweepFullEvaluator(LiveSweepSettings()).evaluate(
        results, 2, "judge", [], DEFAULT_OBJECTIVE_WEIGHTS, None, run
    )
    assert [r["config"]["top_k"] for r in ranked] == [3, 5, 2]
    assert ranked[0]["composite_score"] == pytest.approx(0.12)
    assert ranked[-1]["composite_score"] is None
    assert ranked[-1]["retrieval_score"] == 0.8


@pytest.mark.asyncio
async def test_incomplete_config_is_excluded_instead_of_reweighting():
    results = [{"config": {}, "composite_score": score} for score in (1, 0.9)]
    run = AsyncMock(
        side_effect=[
            {"hit_rate": 1, "avg_faithfulness": 10},
            {"hit_rate": 0, "avg_faithfulness": 1, "avg_relevancy": 1},
        ]
    )
    ranked = await SweepFullEvaluator(LiveSweepSettings()).evaluate(
        results, 2, "judge", [], DEFAULT_OBJECTIVE_WEIGHTS, None, run
    )
    assert ranked[0]["retrieval_score"] == 0.9
    assert ranked[-1]["composite_score"] is None


@pytest.mark.asyncio
async def test_no_complete_finalist_fails_without_retrieval_winner():
    run = AsyncMock(return_value={"hit_rate": 1, "avg_faithfulness": 10})
    with pytest.raises(ValueError, match="No fully evaluated"):
        await SweepFullEvaluator(LiveSweepSettings()).evaluate(
            [{"config": {}, "composite_score": 1}], 1, "judge", [], DEFAULT_OBJECTIVE_WEIGHTS, None, run
        )
