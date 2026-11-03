"""Protect experiments from invented quotes and repeated paid calls."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest


if TYPE_CHECKING:
    from types import ModuleType


@pytest.fixture
def experiment(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / "scripts" / "benchmark"))
    path = Path(__file__).parents[2] / "scripts" / "benchmark/procedure_evidence_experiment.py"
    spec = importlib.util.spec_from_file_location("procedure_experiment", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def table(experiment: ModuleType) -> str:
    row = dict.fromkeys(experiment.FIELDS, "не указано")
    row.update({"этап": "хранение", "документ": "ТТН-1", "источник": "[1]", "цитата": "При передаче"})
    return json.dumps({"rows": [row]}, ensure_ascii=False)


def test_invented_quote_or_source_is_rejected(experiment: ModuleType) -> None:
    text = table(experiment)
    assert experiment.parse_table(text, "[1] При передаче")
    with pytest.raises(ValueError, match="quote"):
        experiment.parse_table(text, "[1] При отгрузке")
    with pytest.raises(ValueError, match="Unknown source"):
        experiment.parse_table(text, "[2] При передаче")


@pytest.mark.asyncio
async def test_resume_reuses_extraction_after_generation_failure(
    experiment: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(experiment.settings, "openrouter_api_key", "fixture")
    snapshot = {
        "model": "fixture-model",
        "seeds": [4101],
        "cases": [
            {
                "id": 154,
                "question": "Порядок?",
                "context": "[1] При передаче",
                "reference_answer": "gold-never-sent",
            }
        ],
    }
    calls = []

    async def complete(
        client: object, messages: list[dict[str, str]], model: str, seed: int, json_output: bool = False
    ) -> dict[str, object]:
        assert "gold-never-sent" not in json.dumps(messages)
        calls.append(json_output)

        if len(calls) == 2:
            message = "generation interrupted"
            raise TimeoutError(message)

        return {"text": table(experiment) if json_output else "answer"}

    monkeypatch.setattr(experiment, "completion", complete)
    out = tmp_path / "result.json"
    with pytest.raises(TimeoutError):
        await experiment.run(snapshot, out)
    result = await experiment.run(snapshot, out)
    assert calls == [True, False, False]
    assert result["results"][0]["generation"]["text"] == "answer"
    snapshot["model"] = "changed"
    with pytest.raises(ValueError, match="Resume input mismatch"):
        await experiment.run(snapshot, out)
