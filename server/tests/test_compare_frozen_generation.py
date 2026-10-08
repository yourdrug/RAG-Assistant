"""Protect paid replay against changed inputs, gold leakage and duplicate requests."""

import importlib.util
import json
from pathlib import Path

import httpx
import pytest


@pytest.fixture
def comparison():
    path = Path(__file__).parents[1] / "scripts" / "compare_frozen_generation.py"
    spec = importlib.util.spec_from_file_location("compare_frozen_generation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def snapshot(comparison):
    run = {
        "id": 10,
        "sweep_id": 6,
        "dataset": "fixture",
        "per_question_results": [
            {
                "id": 148,
                "question": "Какой код?",
                "answer": "old",
                "expected_answer": "reference-never-sent",
                "correctness": 0,
                "breadth": "narrow",
                "evidence_metrics": {"context_fact_coverage": 1},
                "evidence": {
                    "context": "005 | target",
                    "prompt_budget": {"effective_breadth": "narrow", "enumerate_cases": False},
                },
            }
        ],
    }
    return comparison.prepare_snapshot(
        [run],
        [{"id": 148, "annotations": {"required_facts": ["annotation-never-sent"]}}],
        10,
        [148],
        [],
    )


def test_changed_messages_and_context_cannot_be_resumed(comparison, snapshot):
    previous = comparison.initial_results(snapshot)
    snapshot["cases"][0]["messages"][0]["content"] += " changed"
    with pytest.raises(ValueError, match="messages changed"):
        comparison.initial_results(snapshot, previous)


def test_changed_runtime_cannot_reuse_paid_results(comparison, snapshot):
    previous = comparison.initial_results(snapshot)
    snapshot["runtime"]["max_tokens"] += 1
    with pytest.raises(ValueError, match="snapshot mismatch"):
        comparison.initial_results(snapshot, previous)


@pytest.mark.asyncio
async def test_failed_request_resumes_without_repeating_successes(
    comparison, snapshot, tmp_path, monkeypatch
):
    from config import settings

    monkeypatch.setattr(settings, "openrouter_api_key", "fixture-key")
    requests = []
    fail_first = True

    def respond(request):
        nonlocal fail_first
        if request.method == "GET":
            if request.url.path.endswith("/endpoints"):
                return httpx.Response(200, json={"data": {"endpoints": []}})
            return httpx.Response(
                200, json={"data": [{"id": model} for model in snapshot["runtime"]["models"]]}
            )
        payload = json.loads(request.content)
        requests.append(payload)
        if fail_first:
            fail_first = False
            return httpx.Response(429, json={"error": "rate limit"})
        return httpx.Response(
            200,
            json={
                "id": str(len(requests)),
                "model": payload["model"],
                "provider": "fixture",
                "choices": [{"message": {"content": "005"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.001},
            },
        )

    client_class = httpx.AsyncClient
    monkeypatch.setattr(
        comparison.httpx,
        "AsyncClient",
        lambda **kwargs: client_class(transport=httpx.MockTransport(respond), **kwargs),
    )
    out = tmp_path / "results.json"
    with pytest.raises(RuntimeError, match="5/6"):
        await comparison.run_comparison(snapshot, out)
    saved = comparison.read_json(out)
    paid_ids = {row["generation_id"] for row in saved["results"]}
    result = await comparison.run_comparison(snapshot, out)
    assert len(requests) == 7
    assert len(result["results"]) == 6
    assert paid_ids.issubset(row["generation_id"] for row in result["results"])
    assert len({(row["id"], row["requested_model"], row["repeat"]) for row in result["results"]}) == 6
    for payload in requests:
        serialized = json.dumps(payload)
        assert "reference-never-sent" not in serialized
        assert "annotation-never-sent" not in serialized
        assert payload["messages"] == snapshot["cases"][0]["messages"]
    await comparison.run_comparison(snapshot, out)
    assert len(requests) == 7
