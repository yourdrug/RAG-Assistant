"""Frozen generation cases preserve evidence while adding independent reading rules."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from domain.value_objects.llm_provider import Breadth
from infrastructure.ml.rag.rag_prompts import build_prompt

FIXTURE = Path(__file__).parent / "fixtures" / "fixed_generation_reading.json"


@pytest.mark.parametrize("breadth", list(Breadth))
@pytest.mark.parametrize("enumerate_cases", [True, False])
def test_precision_rules_apply_to_short_answers_without_conditional_expansion(breadth, enumerate_cases):
    messages = build_prompt(
        breadth.value,
        enumerate_cases=enumerate_cases,
        question="Кто дополнительно определяет формат поля?",
    ).format_messages(
        context="untrusted evidence",
        history=[],
        question="question",
    )
    system = messages[0].content
    assert "<table_column_reading>" in system
    assert "<norm_subject_reading>" in system
    assert "<editorial_note_reading>" in system
    assert "untrusted evidence" not in system
    assert "untrusted evidence" in messages[1].content


@pytest.mark.asyncio
async def test_fixed_context_replay_sends_unchanged_evidence_without_retrieval(monkeypatch):
    path = Path(__file__).parents[1] / "scripts" / "replay_fixed_generation.py"
    spec = importlib.util.spec_from_file_location("fixed_generation_replay", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    snapshot = json.loads(FIXTURE.read_text())
    model = SimpleNamespace(
        ainvoke=AsyncMock(return_value=SimpleNamespace(content="answer", usage_metadata={}))
    )
    monkeypatch.setattr(module, "ChatOpenAI", lambda **kwargs: model)
    result = await module.replay(snapshot)
    assert [case["id"] for case in result["results"]] == [136, 146, 152]
    for case, call, actual in zip(
        snapshot["cases"], model.ainvoke.call_args_list, result["results"], strict=True
    ):
        messages = call.args[0]
        assert case["context"] in messages[1].content
        assert messages[-1].content.startswith(case["question"])
        assert actual["context_sha256"] == hashlib.sha256(case["context"].encode()).hexdigest()
        assert actual["context_chars"] == len(case["context"])


def test_reading_anchors_quote_exact_evidence_and_bind_note_to_previous_line():
    from infrastructure.ml.rag.evidence_focus import reading_anchors

    snapshot = json.loads(FIXTURE.read_text())
    cases = {case["id"]: case for case in snapshot["cases"]}
    anchors = reading_anchors(cases[146]["context"], cases[146]["question"])
    assert len(anchors) == 1
    assert anchors[0].startswith("судом общей юрисдикции;\n(абзац введен")
    assert "нотариусом" not in anchors[0]
    table = reading_anchors(cases[136]["context"], cases[136]["question"])
    assert table[-1].startswith("| 23 |") and "an..19" in table[-1]
    assert any("an..15" in anchor for anchor in table)
    for question_id in [136, 146]:
        case = cases[question_id]
        for anchor in reading_anchors(case["context"], case["question"]):
            assert all(line in case["context"] for line in anchor.splitlines())


def test_editorial_anchor_requires_requested_date_and_cannot_supply_missing_evidence():
    from infrastructure.ml.rag.evidence_focus import reading_anchors

    context = "wrong subject;\n(абзац введен Законом от 01.01.2020)\nnext subject;"
    assert reading_anchors(context, "Кто дополнительно получил право 16.03.2026?") == []
    assert reading_anchors("", "Кто дополнительно получил право?") == []
