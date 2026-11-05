"""Sweep modes, shortlist diversity and shared A/B evidence scoring."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

from langchain.schema import Document
from pydantic import ValidationError
import pytest

from application.benchmark import compute_summary_from_results
from application.benchmark.sweep_evaluation import SweepEvaluationPlan, SweepFullEvaluator
from application.benchmark.sweep_jobs import SweepJobService
from application.benchmark.sweep_ports import SweepCancelled
from composition.service_providers import build_sweep_engine
from config import get_setting, settings
from domain.benchmark.evaluation import evaluate_retrieval, summarize_evidence
from domain.benchmark.selection import select_fast_candidates
from domain.benchmark.sweep import BenchmarkStrategy, BenchmarkSweep, SweepEvaluationMode
from domain.identity.roles import UserKind, UserRole
from domain.identity.user_context import UserContext
from domain.rag.context import RagContext
from infrastructure.benchmark.evaluation.answer_generators import BenchmarkAnswer
from infrastructure.benchmark.evaluation.case_evaluator import BenchmarkCaseEvaluator
from infrastructure.benchmark.judging import context_judge, evidence_judge, judge
from infrastructure.benchmark.sweep.scoring import (
    compute_composite_score,
    score_config_cheap,
)
from infrastructure.benchmark.sweep.search_space import generate_grid_points
from infrastructure.benchmark.sweep.settings import LiveSweepSettings, settings_snapshot
from infrastructure.ml.rag.benchmark_evidence import BenchmarkEvidence
from presentation.api.helpers import sweep_create_to_dto, sweep_to_response
from presentation.api.schemas.benchmark import SweepCreateRequest
from tests.fakes import FakeUnitOfWorkFactory


if TYPE_CHECKING:
    from domain.benchmark.run import BenchmarkRun


@pytest.fixture(autouse=True)
def individual_judge_mode(monkeypatch: pytest.MonkeyPatch) -> None:

    monkeypatch.setattr(settings, "benchmark_judge_grouped_enabled", False)


def document(text: str, **metadata: Any) -> dict[str, Any]:
    return {"content": text, "metadata": {"source": "law.pdf", **metadata}}


def question() -> dict[str, Any]:
    return {
        "question": "Article 17?",
        "source_hint": "law.pdf",
        "annotations": {"expected_fragments": [{"source": "law.pdf", "text": "article 17"}]},
    }


@pytest.mark.parametrize("mode", list(SweepEvaluationMode))
def test_mode_survives_api_dto_and_response(mode: SweepEvaluationMode) -> None:
    body = SweepCreateRequest(search_space={}, evaluation_mode=mode, top_n_llm=0)
    dto = sweep_create_to_dto(body)
    assert dto.evaluation_mode == mode
    saved = BenchmarkSweep(id=1, evaluation_mode=SweepEvaluationMode(dto.evaluation_mode))
    assert sweep_to_response(saved).evaluation_mode == mode
    assert SweepCreateRequest(search_space={}).evaluation_mode == SweepEvaluationMode.FAST


@pytest.mark.parametrize(
    "strategy",
    [BenchmarkStrategy.RANDOM, BenchmarkStrategy.OPTUNA_TPE],
)
def test_full_mode_rejects_incomplete_search_strategies(strategy: BenchmarkStrategy) -> None:
    with pytest.raises(ValidationError, match="requires grid"):
        SweepCreateRequest(search_space={}, strategy=strategy, evaluation_mode=SweepEvaluationMode.FULL)


def test_tpe_request_preserves_strategy_through_dto_and_response() -> None:
    body = SweepCreateRequest(search_space={}, strategy=BenchmarkStrategy.OPTUNA_TPE)
    dto = sweep_create_to_dto(body)
    assert dto.strategy == BenchmarkStrategy.OPTUNA_TPE
    assert body.model_dump(mode="json")["strategy"] == BenchmarkStrategy.OPTUNA_TPE.value
    saved = BenchmarkSweep(id=1, strategy=BenchmarkStrategy(dto.strategy))
    assert sweep_to_response(saved).strategy == BenchmarkStrategy.OPTUNA_TPE.value


@pytest.mark.parametrize("strategy", ["unknown", "successive_halving"])
def test_unknown_search_strategy_is_rejected(strategy: str) -> None:
    with pytest.raises(ValidationError):
        SweepCreateRequest.model_validate({"search_space": {}, "strategy": strategy})


@pytest.mark.asyncio
@pytest.mark.parametrize("budget", [0, 1, 3])
async def test_full_mode_evaluates_all_25_combinations_and_can_find_last_winner(
    monkeypatch: pytest.MonkeyPatch, budget: int
) -> None:
    runtime = LiveSweepSettings()
    previous = runtime.top_k
    engine = build_sweep_engine(FakeUnitOfWorkFactory(), runtime=runtime)
    snapshot = [{**question(), "id": 17}]
    monkeypatch.setattr(engine, "load_questions", AsyncMock(return_value=snapshot))
    monkeypatch.setattr(engine, "cache_candidates", AsyncMock(return_value=({}, {}, {})))
    monkeypatch.setattr(engine, "score_config_cheap", lambda *_a, **_kw: {"composite_score": 1})
    tested: list[tuple[int, Any]] = []

    async def full(questions: list[dict[str, Any]], model: str, **kwargs: Any) -> dict[str, Any]:

        assert questions is snapshot
        assert model == "judge"
        tested.append((runtime.top_k, get_setting("rerank_min_score")))
        return {"hit_rate": 1, "avg_relevancy": 10 if tested[-1] == (5, 0.5) else 1}

    monkeypatch.setattr(engine, "run_full_benchmark", full)
    sweep = BenchmarkSweep(
        evaluation_mode=SweepEvaluationMode.FULL,
        top_n_llm=budget,
        search_space={
            "top_k": {"values": [1, 2, 3, 4, 5]},
            "rerank_min_score": {"values": [0.1, 0.2, 0.3, 0.4, 0.5]},
        },
        objective_weights={"hit_rate": 0.1, "relevancy": 0.9},
    )
    progress: list[tuple[Any, ...]] = []
    results = await engine.run_sweep(sweep, "judge", lambda *args: progress.append(args))
    assert len(tested) == len(set(tested)) == 25
    assert results[0]["config"] == {"top_k": 5, "rerank_min_score": 0.5}
    assert results[0]["composite_score"] == 1
    assert all(r["llm_evaluated"] and r["evaluated_config_count"] == 25 for r in results)
    assert progress[-1][:2] == (25, 25)
    assert progress[-1][2]["phase"] == "full_evaluation"
    assert runtime.top_k == previous


@pytest.mark.asyncio
async def test_full_mode_requires_judge_before_expensive_preparation() -> None:
    with pytest.raises(ValueError, match="requires a judge"):
        await build_sweep_engine(FakeUnitOfWorkFactory()).run_sweep(
            BenchmarkSweep(evaluation_mode=SweepEvaluationMode.FULL)
        )


@pytest.mark.parametrize("parameter", ["chunk_size", "openrouter_model"])
@pytest.mark.asyncio
async def test_full_mode_rejects_parameters_that_cannot_be_applied(
    monkeypatch: pytest.MonkeyPatch, parameter: str
) -> None:
    engine = build_sweep_engine(FakeUnitOfWorkFactory())
    load = AsyncMock()
    monkeypatch.setattr(engine, "load_questions", load)
    with pytest.raises(ValueError, match="supports retrieval parameters only"):
        await engine.run_sweep(
            BenchmarkSweep(
                evaluation_mode=SweepEvaluationMode.FULL, search_space={parameter: {"values": [1]}}
            ),
            "judge",
        )
    load.assert_not_called()


@pytest.mark.asyncio
async def test_full_mode_can_optimize_only_judge_quality(monkeypatch: pytest.MonkeyPatch) -> None:
    body = SweepCreateRequest(
        search_space={"top_k": {"values": [1, 2]}},
        evaluation_mode=SweepEvaluationMode.FULL,
        objective_weights={"faithfulness": 1},
    )
    engine = build_sweep_engine(FakeUnitOfWorkFactory())
    monkeypatch.setattr(engine, "load_questions", AsyncMock(return_value=[{"question": "unlabelled"}]))
    monkeypatch.setattr(engine, "cache_candidates", AsyncMock(return_value=({}, {}, {})))
    run = AsyncMock(side_effect=[{"avg_faithfulness": 2}, {"avg_faithfulness": 9}])
    monkeypatch.setattr(engine, "run_full_benchmark", run)
    sweep = BenchmarkSweep(
        search_space=body.search_space,
        evaluation_mode=body.evaluation_mode,
        objective_weights=body.objective_weights,
    )
    results = await engine.run_sweep(sweep, "judge")
    assert run.await_count == 2
    assert results[0]["config"]["top_k"] == 2
    assert results[0]["composite_score"] == pytest.approx(0.9)


def test_fast_shortlist_uses_fragment_completeness_and_mrr_for_ties() -> None:
    results = [
        {"config": {"top_k": 1}, "composite_score": 1, "avg_fragment_recall_at_k": 0.5, "avg_mrr": 1},
        {"config": {"top_k": 2}, "composite_score": 1, "avg_fragment_recall_at_k": 1, "avg_mrr": 0.5},
        {"config": {"top_k": 3}, "composite_score": 1, "avg_fragment_recall_at_k": 1, "avg_mrr": 1},
    ]
    assert select_fast_candidates(results, 1)[0] is results[2]
    assert [r["config"] for r in results] == [{"top_k": k} for k in (1, 2, 3)]


def test_fast_shortlist_keeps_elite_and_diverse_competitive_parameters() -> None:
    results = [{"config": {"top_k": k}, "composite_score": 1} for k in (1, 2, 3, 4, 20)]
    selected = select_fast_candidates(results, 3)
    assert [r["config"]["top_k"] for r in selected] == [1, 2, 20]
    assert select_fast_candidates(results, 0) == []


def test_diversity_does_not_prefer_a_bad_config_and_duplicates_do_not_use_budget() -> None:
    results = [
        {"config": {"top_k": 1}, "composite_score": 1},
        {"config": {"top_k": 1}, "composite_score": 1},
        {"config": {"top_k": 2}, "composite_score": 0.95},
        {"config": {"top_k": 3}, "composite_score": 0.94},
        {"config": {"top_k": 100}, "composite_score": 0},
    ]
    assert [r["config"]["top_k"] for r in select_fast_candidates(results, 3)] == [1, 2, 3]


@pytest.mark.asyncio
async def test_full_mode_can_be_cancelled_and_restores_settings() -> None:
    runtime = LiveSweepSettings()
    previous = runtime.top_k
    run = AsyncMock(return_value={"hit_rate": 1})
    results = [{"config": {"top_k": k}, "composite_score": 1} for k in (1, 2, 3)]
    with pytest.raises(SweepCancelled):
        await SweepFullEvaluator(runtime).evaluate(
            results,
            SweepEvaluationPlan(1, "judge", {"hit_rate": 1}, SweepEvaluationMode.FULL),
            [],
            AsyncMock(side_effect=[False, True]),
            run,
        )
    assert run.await_count == 1
    assert runtime.top_k == previous


@pytest.fixture
def mock_judges(monkeypatch: pytest.MonkeyPatch) -> None:

    monkeypatch.setattr(
        judge,
        "judge_answer_async",
        AsyncMock(return_value={"faithfulness": 10, "relevancy": 10, "correctness": 10}),
    )
    monkeypatch.setattr(context_judge, "compute_context_precision_recall", lambda *_a, **_k: {})
    monkeypatch.setattr(evidence_judge, "judge_evidence", lambda *_a, **_kw: {"scores": {}, "details": {}})


@pytest.mark.asyncio
@pytest.mark.parametrize("source_hint", [None, "law.pdf"])
@pytest.mark.parametrize("text,expected", [("article 29", 0), ("article 17", 1)])
async def test_a_and_b_share_fragment_scoring_even_when_legacy_source_is_a_hit(
    mock_judges: None, text: str, expected: float, source_hint: str | None
) -> None:
    case = question()
    case["source_hint"] = source_hint
    doc = Document(page_content=text, metadata={"source": "law.pdf"})
    cheap = score_config_cheap(
        {"top_k": 1, "fetch_k": 1},
        [case],
        {case["question"]: [("h", 1, doc)]},
        {},
        {"h": doc},
        weights={"hit_rate": 1},
        defaults=settings_snapshot(),
    )
    evidence = BenchmarkEvidence([document(text)], [document(text)], text)
    answer = BenchmarkAnswer(text, text, {"hit_rate": 1, "mrr": 1}, 1, 1, evidence=evidence)
    evaluated = await BenchmarkCaseEvaluator(
        SimpleNamespace(generate=AsyncMock(return_value=answer)), "judge"
    ).evaluate(
        1,
        case,
        1,
        RagContext(user=UserContext(user_id=1, user_kind=UserKind.INTERNAL, user_role=UserRole.USER)),
    )
    summary = compute_summary_from_results([evaluated])
    assert cheap["avg_hit_rate"] == summary["hit_rate"] == expected
    assert cheap["avg_mrr"] == summary["avg_mrr"] == expected
    assert summary["avg_source_hit_rate"] == (1 if source_hint else None)
    assert summary["retrieval_annotated_count"] == 1
    assert summary["avg_fragment_recall_at_k"] == expected


def test_grid_includes_float_thresholds_without_truncation_or_rounding_drift() -> None:
    points = generate_grid_points(
        {
            "rerank_min_score": {"min": 0.1, "max": 0.3, "step": 0.1},
            "top_k": {"min": 1, "max": 2},
        }
    )
    assert len(points) == 6
    assert [p["rerank_min_score"] for p in points] == [0.1, 0.1, 0.2, 0.2, 0.3, 0.3]


@pytest.mark.parametrize(
    "spec",
    [
        {"min": 1, "max": 2, "step": 0},
        {"min": 2, "max": 1},
        {"min": 0, "max": float("inf")},
    ],
)
def test_invalid_grid_ranges_are_rejected(spec: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="Grid ranges"):
        generate_grid_points({"top_k": spec})


@pytest.mark.asyncio
async def test_prompt_loss_lowers_context_objectives_and_final_score(mock_judges: None) -> None:
    case = question()
    case["annotations"]["required_facts"] = ["30 days"]
    docs = [document("article 17"), document("30 days")]

    async def evaluate(selected: list[dict[str, Any]], context: str) -> dict[str, Any]:
        evidence = BenchmarkEvidence(docs, selected, context)
        answer = BenchmarkAnswer("a", context, {"hit_rate": 1, "mrr": 1}, 1, 1, evidence=evidence)
        return await BenchmarkCaseEvaluator(
            SimpleNamespace(generate=AsyncMock(return_value=answer)), "judge"
        ).evaluate(
            1,
            case,
            1,
            RagContext(user=UserContext(user_id=1, user_kind=UserKind.INTERNAL, user_role=UserRole.USER)),
        )

    lost, complete = await evaluate([], "other article"), await evaluate(docs, "article 17 30 days")
    assert lost["retriever_metrics"]["mrr"] == 1
    assert lost["retriever_metrics"]["evidence_completion_rr"] == 0.5
    assert lost["evidence_metrics"]["context_fragment_recall"] == 0
    assert lost["evidence_metrics"]["context_fact_coverage"] == 0
    summaries = [compute_summary_from_results([r]) for r in (lost, complete)]
    run = AsyncMock(side_effect=summaries)
    results = [{"config": {"top_k": k}, "composite_score": 1} for k in (1, 2)]
    ranked = await SweepFullEvaluator(LiveSweepSettings()).evaluate(
        results,
        SweepEvaluationPlan(
            2,
            "judge",
            {"fragment_recall_at_k": 0.2, "context_fragment_recall": 0.4, "context_fact_coverage": 0.4},
        ),
        [case],
        None,
        run,
    )
    assert ranked[0]["config"]["top_k"] == 2
    assert ranked[0]["composite_score"] == 1
    assert ranked[1]["composite_score"] == pytest.approx(0.2)


@pytest.mark.parametrize("documents", [None, []])
def test_missing_evidence_is_distinct_from_retrieval_miss(documents: list[dict[str, Any]] | None) -> None:
    metrics = evaluate_retrieval(question(), documents)
    assert metrics["hit_rate"] == (None if documents is None else 0)
    assert metrics["fragment_mrr"] == (None if documents is None else 0)


def test_fact_completion_is_distinct_from_first_fragment_rank() -> None:
    case = question()
    case["annotations"]["required_facts"] = ["filing", ["30 days", "one month"]]
    metrics = evaluate_retrieval(case, [document("article 17 filing"), document("one month")])
    assert metrics["hit_rate"] == 1
    assert metrics["mrr"] == 1
    assert metrics["evidence_completion_rr"] == 0.5
    assert metrics["retrieval_fact_coverage"] == 1


def test_fragment_objectives_have_explicit_normalized_weights() -> None:
    request = SweepCreateRequest(
        search_space={}, objective_weights={"fragment_recall_at_k": 0.5, "context_fact_coverage": 0.5}
    )
    assert (
        compute_composite_score(
            {"fragment_recall_at_k": 1, "context_fact_coverage": 0}, request.objective_weights
        )
        == 0.5
    )
    assert all(
        weight == 0
        for key, weight in request.objective_weights.items()
        if key not in {"fragment_recall_at_k", "context_fact_coverage"}
    )


@pytest.mark.asyncio
async def test_partial_missing_context_cannot_improve_composite_by_excluding_cases() -> None:
    cases = [
        {"annotations": {"required_facts": ["fact"]}, "evidence_metrics": {"context_fact_coverage": value}}
        for value in (1, None)
    ]
    full = {"hit_rate": 1, **summarize_evidence(cases)}
    results = await SweepFullEvaluator(LiveSweepSettings()).evaluate(
        [{"config": {}, "composite_score": 1}],
        SweepEvaluationPlan(1, "judge", {"hit_rate": 0.1, "context_fact_coverage": 0.9}),
        [],
        None,
        AsyncMock(return_value=full),
    )
    assert results[0]["composite_score"] is None
    assert results[0]["evaluation_complete"] is False


@pytest.mark.asyncio
async def test_worker_saves_evidence_scores_and_selection_scope(monkeypatch: pytest.MonkeyPatch) -> None:

    factory = FakeUnitOfWorkFactory()
    sweep = BenchmarkSweep(evaluation_mode=SweepEvaluationMode.FULL)
    created: list[BenchmarkRun] = []

    async def create(run: BenchmarkRun) -> BenchmarkRun:
        run.id = 3
        created.append(run)
        return run

    monkeypatch.setattr(factory._uow.benchmark_runs, "create", create)
    monkeypatch.setattr(factory._uow.benchmark_sweeps, "set_best_run", AsyncMock())
    results = [
        {
            "config": {"top_k": 2},
            "composite_score": 1,
            "llm_evaluated": True,
            "evaluated_config_count": 1,
            "full_metrics": {
                "hit_rate": 1,
                "avg_fragment_recall_at_k": 1,
                "avg_context_fact_coverage": 1,
                "fragment_recall_at_k_evaluated_count": 7,
                "retrieval_annotated_count": 7,
            },
        }
    ]
    assert await SweepJobService(factory, AsyncMock(), AsyncMock()).save_sweep_results(sweep, 1, results) == 3
    assert created[0].summary_metrics["avg_context_fact_coverage"] == 1
    assert created[0].summary_metrics["evaluation_mode"] == SweepEvaluationMode.FULL
    assert created[0].summary_metrics["fragment_recall_at_k_evaluated_count"] == 7
