"""Production prompt assembly for unchanged fields, table pairs and complete lists."""

import json
from pathlib import Path

import pytest

from domain.services.rag_policy.evidence_reading import evidence_reading_reminder
from infrastructure.ml.rag.evidence_focus import (
    enumeration_anchors,
    format_generation_messages,
    table_lookup_anchors,
)
from infrastructure.ml.rag.rag_prompts import build_prompt

SNAPSHOT = json.loads(
    (Path(__file__).parent / "fixtures" / "generation_completeness_reading.json").read_text()
)


@pytest.mark.parametrize("case", SNAPSHOT["cases"], ids=lambda case: str(case["id"]))
@pytest.mark.parametrize("enumerate_cases", [False, True])
def test_reading_checks_reach_model_after_frozen_context(case, enumerate_cases):
    prompt = build_prompt(question=case["question"], enumerate_cases=enumerate_cases)
    messages = format_generation_messages(
        prompt, context=case["context"], history=[], question=case["question"]
    )
    assert case["context"] in messages[1].content
    assert case["context"] not in messages[0].content
    assert messages[-1].content == case["question"] + evidence_reading_reminder(case["question"])
    system = messages[0].content
    reminder = messages[-1].content
    if case["id"] == 140:
        assert "не изменилось" in system and "не изменилось" in reminder
        assert "одному полю (номер И название)" in reminder
        assert "подтверждённым исходной и запрошенной редакциям" in reminder
        assert "не делай вывод о неизменности из отсутствия данных" in reminder
        assert "с отличающимся форматом" not in reminder
    elif case["id"] == 148:
        # This question asks for a code without mentioning a table or field.
        assert "<table_column_reading>" in system
        assert "название строки → значение" in reminder
        assert "Отклони неподтверждённую пару" in reminder
        assert "значение соседней строки не подходит" in reminder
        assert "ведущие нули" in reminder
    else:
        assert "<enumeration_reading>" in system
        assert "из всех релевантных фрагментов" in reminder
        assert "ни одна категория, этап, условие или исключение" in reminder
        assert "не добавляй" in reminder.lower()


def test_unrelated_question_does_not_receive_reading_checklist():
    assert evidence_reading_reminder("Где скачать приложение?") == ""


def test_code_lookup_binds_value_to_target_row_instead_of_neighbor():
    case = next(case for case in SNAPSHOT["cases"] if case["id"] == 148)
    anchors = table_lookup_anchors(case["context"], case["question"])
    assert len(anchors) == 1
    assert "| 005 | Донецкая область | Украина |" in anchors[0]
    assert "012" not in anchors[0]
    assert "Наименование административно-территориального деления" in anchors[0]
    assert all(line in case["context"] for line in anchors[0].splitlines())
    assert table_lookup_anchors("", case["question"]) == []


def test_code_lookup_keeps_ambiguous_rows_for_model_to_check():
    context = "| 001 | Северная область | Страна А |\n| 002 | Северная область | Страна Б |"
    assert len(table_lookup_anchors(context, "Какой код Северной области?")) == 2


@pytest.mark.parametrize("question_id", [154, 155])
def test_enumeration_focus_preserves_all_saved_cases_and_exact_text(question_id):
    case = next(case for case in SNAPSHOT["cases"] if case["id"] == question_id)
    anchors = enumeration_anchors(case["context"], case["question"])
    assert all(anchor in case["context"] for anchor in anchors)
    focus = "\n".join(anchors)
    if question_id == 154:
        for phrase in ["При передаче", "При отгрузке", "письменного указания", "ТН-2", "стоимостных"]:
            assert phrase in focus
    else:
        for phrase in ["потребительской", "отсутствии", "групповой", "набора", "комплекта"]:
            assert phrase in focus
        assert "знак защиты на незащищенный" not in focus  # belongs to 5.3, not requested 5.1
    messages = format_generation_messages(
        build_prompt(question=case["question"]),
        context=case["context"],
        history=[],
        question=case["question"],
    )
    assert all(anchor in messages[-2].content for anchor in anchors)
    assert messages[-2].type == "human"


def test_enumeration_focus_cannot_supply_missing_or_wrong_subpoint_evidence():
    question = "Какие случаи предусмотрены п. 5.1?"
    assert enumeration_anchors("", question) == []
    assert (
        enumeration_anchors(
            "[Раздел: point 5 › subpoint_num 5.10]\nв упаковке - неверный подпункт;", question
        )
        == []
    )
