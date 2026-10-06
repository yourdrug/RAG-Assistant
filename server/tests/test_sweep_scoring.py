"""Characterization and regression contracts for sweep scoring and final selection."""

from unittest.mock import AsyncMock
from types import SimpleNamespace

import pytest
from langchain.schema import Document
from pydantic import ValidationError

from config import get_setting
from domain.value_objects.benchmark_scoring import DEFAULT_OBJECTIVE_WEIGHTS
from domain.entities.benchmark_sweep import BenchmarkSweep
from domain.value_objects.benchmark_strategy import BenchmarkStrategy
from infrastructure.benchmark.sweep_engine import SweepEngine
from infrastructure.benchmark.sweep_full_evaluation import SweepFullEvaluator
from infrastructure.benchmark.sweep_scoring import compute_composite_score, score_config_cheap
from infrastructure.benchmark.sweep_settings import LiveSweepSettings
from presentation.api.schemas.benchmark import SweepCreateRequest


def score_documents(documents, annotations=None, source_hint="law.pdf", rerank_scores=None, **config):
    return score_config_cheap(
        {"top_k": 5, "fetch_k": 10, "dense_weight": 1, "sparse_weight": 0, **config},
        [{"question": "article 17?", "source_hint": source_hint, "annotations": annotations}],
        {"article 17?": [(str(i), 1, doc) for i, doc in enumerate(documents)]},
        {},
        {str(i): doc for i, doc in enumerate(documents)},
        {"hit_rate": 0.5, "mrr": 0.5},
        rerank_scores=rerank_scores,
    )


@pytest.mark.parametrize("top_k, expected", [(1, 0), (2, 0.5)])
def test_cheap_scoring_characterizes_top_k_and_first_source_rank(top_k, expected):
    metrics = score_documents(
        [
            Document(page_content="other", metadata={"filename": "other.pdf"}),
            Document(page_content="article 29", metadata={"filename": "law.pdf"}),
        ],
        top_k=top_k,
    )
    assert metrics["avg_mrr"] == expected
    assert metrics["avg_hit_rate"] == int(expected > 0)


def test_threshold_comparison_requires_cached_reranker_scores():
    with pytest.raises(ValueError, match="Cached reranker scores"):
        score_documents([], rerank_min_score=0.1)


@pytest.mark.parametrize("correct_rank", [None, 2, 5])
def test_fragment_annotation_rejects_wrong_article_from_correct_law(correct_rank):
    docs = [Document(page_content="article 29", metadata={"filename": "law.pdf"}) for _ in range(5)]
    if correct_rank:
        docs[correct_rank - 1] = Document(page_content="article 17", metadata={"filename": "law.pdf"})
    metrics = score_documents(docs, {"expected_fragments": [{"source": "law.pdf", "text": "article 17"}]})
    assert metrics["avg_hit_rate"] == int(correct_rank is not None)
    assert metrics["avg_mrr"] == (1 / correct_rank if correct_rank else 0)


@pytest.mark.parametrize("metadata", [{"section": "29", "page": 1}, {"section": "17", "page": 2}])
def test_fragment_scoring_respects_section_and_pages(metadata):
    metrics = score_documents(
        [Document(page_content="article 17", metadata={"source": "/docs/law.pdf", **metadata})],
        {"expected_fragments": [{"source": "law.pdf", "text": "article 17", "section": "17", "pages": [1]}]},
    )
    assert metrics["avg_hit_rate"] == 0


@pytest.mark.parametrize("include_second_fact", [False, True])
def test_required_facts_must_be_covered_across_retrieved_chunks(include_second_fact):
    docs = [
        Document(page_content="Article 17 requires filing", metadata={"filename": "law.pdf"}),
        Document(
            page_content="within 30 days" if include_second_fact else "article 29",
            metadata={"filename": "law.pdf"},
        ),
    ]
    metrics = score_documents(docs, {"required_facts": ["requires filing", ["within 30 days", "one month"]]})
    assert metrics["avg_hit_rate"] == int(include_second_fact)
    assert metrics["avg_mrr"] is None  # Facts alone do not label individual fragment ranks.
    assert metrics["avg_evidence_completion_rr"] == (0.5 if include_second_fact else 0)


def test_fragment_hit_does_not_hide_missing_required_fact():
    metrics = score_documents(
        [Document(page_content="article 17", metadata={"filename": "law.pdf"})],
        {"expected_fragments": [{"source": "law.pdf", "text": "article 17"}], "required_facts": ["30 days"]},
    )
    assert metrics["avg_hit_rate"] == 0
    assert metrics["avg_mrr"] == 1  # First relevant fragment, separate from fact completeness.
    assert metrics["avg_evidence_completion_rr"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("top_n", [0, 1])
@pytest.mark.parametrize("parameter", ["rerank_min_score", "rerank_score_gap_ratio"])
async def test_reranker_scores_all_combinations_but_llm_only_scores_shortlist(monkeypatch, top_n, parameter):
    wrong = Document(page_content="article 29", metadata={"source": "law.pdf"})
    correct = Document(page_content="article 17", metadata={"source": "law.pdf", "section": "17"})
    predict = AsyncMock(return_value=[0.9, 0.4])
    engine = SweepEngine(
        uow_factory=None, ml_clients=SimpleNamespace(reranker=lambda: SimpleNamespace(predict=predict))
    )
    question = {
        "question": "q",
        "annotations": {"expected_fragments": [{"source": "law.pdf", "text": "article 17"}]},
    }
    monkeypatch.setattr(engine, "load_questions", AsyncMock(return_value=[question]))
    cache = AsyncMock(
        return_value=(
            {"q": [("wrong", 1, wrong), ("correct", 0.5, correct)]},
            {},
            {"wrong": wrong, "correct": correct},
        )
    )
    monkeypatch.setattr(engine, "cache_candidates", cache)
    thresholds = []

    async def evaluate_current_settings(questions, judge_model):
        thresholds.append(get_setting(parameter))
        return {"hit_rate": 1, "avg_mrr": 0.5}

    previous_threshold = get_setting(parameter)
    run = AsyncMock(side_effect=evaluate_current_settings)
    monkeypatch.setattr(engine, "run_full_benchmark", run)
    other = "rerank_score_gap_ratio" if parameter == "rerank_min_score" else "rerank_min_score"
    sweep = BenchmarkSweep(
        search_space={parameter: {"values": [0.9, 0.1]}, other: {"values": [None]}, "top_k": {"values": [2]}},
        objective_weights={"hit_rate": 1},
        top_n_llm=top_n,
    )
    results = await engine.run_sweep(sweep, judge_model="judge" if top_n else None)
    assert cache.call_args.args[0] == [question]
    assert len(results) == 2
    assert run.await_count == top_n
    assert sum(bool(result.get("llm_evaluated")) for result in results) == top_n
    assert results[0]["config"][parameter] == 0.1
    assert results[0]["avg_hit_rate"] == 1
    assert results[0]["avg_mrr"] == 0.5
    assert results[1]["avg_hit_rate"] == 0
    assert all(call.args[0] == [question] for call in run.call_args_list)
    assert thresholds == ([0.1] if top_n else [])
    assert get_setting(parameter) == previous_threshold
    predict.assert_awaited_once_with([("q", "[law.pdf] article 29"), ("q", "[law.pdf] (17) article 17")])


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", [BenchmarkStrategy.RANDOM, BenchmarkStrategy.OPTUNA_TPE])
async def test_sampled_reranker_trials_respect_llm_budget(monkeypatch, strategy):
    engine = SweepEngine(uow_factory=None)
    monkeypatch.setattr(
        engine, "load_questions", AsyncMock(return_value=[{"question": "q", "source_hint": "law.pdf"}])
    )
    monkeypatch.setattr(engine, "cache_candidates", AsyncMock(return_value=({}, {}, {})))
    run = AsyncMock(return_value={"hit_rate": 1})
    monkeypatch.setattr(engine, "run_full_benchmark", run)
    sweep = BenchmarkSweep(
        strategy=strategy.value,
        search_space={"rerank_min_score": {"values": [0.1, 0.9]}, "_n_random": 3, "_n_trials": 3},
        objective_weights={"hit_rate": 1},
        top_n_llm=1,
    )
    results = await engine.run_sweep(sweep, judge_model="judge")
    assert len(results) == 3
    assert run.await_count == 1


@pytest.mark.parametrize("parameter", ["rerank_min_score", "rerank_score_gap_ratio"])
def test_reranking_precedes_top_k_and_filters(parameter):
    docs = [
        Document(page_content="article 29", metadata={"source": "law.pdf"}),
        Document(page_content="article 17", metadata={"source": "law.pdf"}),
    ]
    annotations = {"expected_fragments": [{"source": "law.pdf", "text": "article 17"}]}
    scores = {"article 17?": {"0": 0.2, "1": 0.8}}
    metrics = score_documents(
        docs, annotations, rerank_scores=scores, top_k=1, rerank_min_score=None, rerank_score_gap_ratio=None
    )
    assert metrics["avg_hit_rate"] == metrics["avg_mrr"] == 1
    config = {"rerank_min_score": None, "rerank_score_gap_ratio": None, parameter: 1.1}
    assert score_documents(docs, annotations, rerank_scores=scores, top_k=1, **config)["avg_hit_rate"] == 0


def test_reranking_respects_fetch_k_candidate_pool():
    docs = [
        Document(page_content="article 29", metadata={"source": "law.pdf"}),
        Document(page_content="article 17", metadata={"source": "law.pdf"}),
    ]
    metrics = score_documents(
        docs,
        {"expected_fragments": [{"source": "law.pdf", "text": "article 17"}]},
        rerank_scores={"article 17?": {"0": 0.2, "1": 0.8}},
        fetch_k=1,
        top_k=1,
        rerank_min_score=None,
        rerank_score_gap_ratio=None,
    )
    assert metrics["avg_hit_rate"] == 0


def test_phase_a_uses_effective_runtime_threshold_defaults():
    docs = [Document(page_content="article 17", metadata={"source": "law.pdf"})]
    annotations = {"expected_fragments": [{"source": "law.pdf", "text": "article 17"}]}
    scores = {"article 17?": {"0": 0.4}}
    runtime = LiveSweepSettings()
    with runtime.override({"rerank_min_score": 0.5, "rerank_score_gap_ratio": None}):
        assert score_documents(docs, annotations, rerank_scores=scores)["avg_hit_rate"] == 0
    with runtime.override({"rerank_min_score": None, "rerank_score_gap_ratio": None}):
        assert score_documents(docs, annotations, rerank_scores=scores)["avg_hit_rate"] == 1


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
    results = [{"config": {"top_k": i}, "composite_score": score} for i, score in enumerate((1, 0.9))]
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
