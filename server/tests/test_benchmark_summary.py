"""Worker and persisted summaries agree on costs and shared metrics."""

from typing import Any

import pytest

from application.benchmark import compute_summary_from_results
from infrastructure.benchmark.metrics import compute_summary_metrics


@pytest.mark.parametrize("observed_cost", [0.02, 0.0, None])
def test_summary_cost_semantics_and_shared_metrics(observed_cost: float | None) -> None:
    results: list[dict[str, Any]] = [
        {
            "id": 1,
            "question": "q",
            "answer": "a",
            "expected_answer": "a",
            "latency_sec": 2,
            "rag_latency_sec": 1,
            "retriever_metrics": {"hit_rate": 1, "mrr": 0.5, "avg_similarity": 0.8},
            "generator_metrics": {"faithfulness": 8, "relevancy": 7, "correctness": 9},
            "input_tokens": 100,
            "output_tokens": 20,
            "judge_input_tokens": 50,
            "judge_output_tokens": 10,
            "judge_calls": 1,
            "cost_usd": observed_cost,
            "estimated_generation_cost_usd": 0.01,
            "breadth": "narrow",
        },
        {
            "id": 2,
            "question": "legacy question",
            "answer": "a",
            "latency_sec": 3,
            "retriever_metrics": {"hit_rate": None, "mrr": None, "avg_similarity": 0},
            "generator_metrics": {"faithfulness": None, "relevancy": None, "correctness": None},
            "cost_usd": 0.003,
        },
    ]
    persisted = compute_summary_metrics(results)
    worker = compute_summary_from_results(results)
    assert {key: worker[key] for key in persisted} == persisted
    assert worker["estimated_cost_usd"] == 0.01
    assert worker["cost_complete"] is (observed_cost is not None)
    assert worker["total_cost_usd"] == (None if observed_cost is None else round(observed_cost + 0.003, 6))
    assert worker["known_cost_usd"] == round((observed_cost or 0) + 0.003, 6)
    assert worker["total_judge_calls"] == 1
    assert worker["breadth_distribution"] == {"narrow": 1, "broad": 0}
    assert [result["id"] for result in worker["results"]] == [1, 2]


def test_empty_summary_has_no_cost() -> None:
    for summarize in (compute_summary_metrics, compute_summary_from_results):
        summary = summarize([])
        assert summary["total_questions"] == 0
        assert summary["estimated_cost_usd"] == 0
        assert summary["total_cost_usd"] == 0
        assert summary["cost_complete"] is True
