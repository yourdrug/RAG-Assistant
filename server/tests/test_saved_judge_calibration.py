"""Frozen judging criteria and paid-call resume; semantic calibration is a real LLM experiment."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Self

import pytest


if TYPE_CHECKING:
    from types import ModuleType


@pytest.fixture
def experiment() -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / "benchmark/calibrate_saved_judge.py"
    spec = importlib.util.spec_from_file_location("saved_judge_calibration", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_exact_criteria_and_resume_preserve_frozen_answers(
    experiment: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = []

    class Client:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> Self:
            return self

        async def __aexit__(self, *args: object) -> None:
            pass

        async def post(self, url: object, *, json: dict[str, Any]) -> SimpleNamespace:
            _ = url
            prompt = json["messages"][0]["content"]
            calls.append(prompt)
            payload = experiment.json.loads(prompt.split("Данные JSON:\n")[1])

            if "answer" in payload:
                assert payload["answer"] == "saved answer"

            metrics = [m for m in experiment.INSTRUCTIONS if f"{m}:" in prompt]

            for m in metrics:
                assert experiment.INSTRUCTIONS[m] in prompt

            if "faithfulness" in metrics:
                assert "expected_answer" not in payload
                assert payload["context"] == "saved context"

            data = {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": experiment.json.dumps(
                                {m: {"score": 10, "reason": "fixture"} for m in metrics}
                            )
                        },
                    }
                ],
                "model": "fixed",
            }
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: data)

    monkeypatch.setattr(experiment.httpx, "AsyncClient", Client)
    snapshot: dict[str, Any] = {
        "judge_model": "fixed",
        "cases": [
            {
                "key": "case",
                "cohort": "calibration",
                "repeats": 3,
                "question": {"question": "?", "expected_answer": "reference", "annotations": {}},
                "answer": "saved answer",
                "context": "saved context",
            }
        ],
    }
    out = tmp_path / "results.json"
    await experiment.run(snapshot, out, "calibration")
    assert len(calls) == 9
    await experiment.run(snapshot, out, "calibration")
    assert len(calls) == 9
    assert len(json.loads(out.read_text())["requests"]) == 9
    snapshot["cases"][0]["context"] = "changed context"
    with pytest.raises(ValueError, match="Resume input/protocol mismatch"):
        await experiment.run(snapshot, out, "calibration")
    assert len(calls) == 9

    snapshot["cases"][0]["context"] = "saved context"
    monkeypatch.setattr(experiment, "JUDGE_RUBRIC_VERSION", "changed-rubric")
    with pytest.raises(ValueError, match="Resume input/protocol mismatch"):
        await experiment.run(snapshot, out, "calibration")
    assert len(calls) == 9
