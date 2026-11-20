"""A question whether an act changed transmission timing compares the amendment with the timing norm."""

from __future__ import annotations

from langchain_core.documents import Document
import pytest

from domain.rag.evidence.assessment import EvidenceStatus
from infrastructure.domain_packs.belarus_law.evidence.applicability import timing_change_question
from infrastructure.domain_packs.belarus_law.evidence.grounded_response import (
    extractive_timing_response,
    scoped_evidence_response,
)
from infrastructure.domain_packs.belarus_law.evidence.reading import has_reading_rule


pytestmark = pytest.mark.belarus

QUESTION = (
    "Изменилась ли периодичность передачи сведений EDI-провайдером в ПК СПТ постановлением "
    "НАН/Минфина/МНС/Минсвязи от 24.09.2024 N 11/49/30/26?"
)
AMENDMENT = Document(
    page_content='после позиции\n| "50 | Код товара по ТН ВЭД ЕАЭС <br> an..10 | М | <ItemCustomCode> |',
    metadata={
        "doc_title": "Постановление от 24 сентября 2024 г. № 11/49/30/26",
        "source": "Постановление 24 сентября 2024 г. N 11493026.docx",
        "document_id": 24,
        "citation_id": 1,
    },
)
NORM = Document(
    page_content=(
        "2.2. сведения, указанные в пункте 1 настоящего постановления:\n"
        "передаются EDI-провайдером грузоотправителя в режиме реального времени в Министерство "
        "по налогам и сборам в виде электронного документа;"
    ),
    metadata={
        "doc_title": "Постановление от 9 апреля 2021 г. № 11 «Список документов»",
        "source": "Постановление МНС от 09.04.2021 N 11.docx",
        "document_id": 25,
        "citation_id": 2,
    },
)


def test_change_question_is_told_apart_from_a_timing_question() -> None:
    assert timing_change_question(QUESTION)
    assert has_reading_rule(QUESTION, "amendment_subject")
    assert not timing_change_question("Как часто передаются сведения согласно постановлению МНС N 11?")
    assert not timing_change_question("Изменилась ли периодичность передачи сведений?")  # No act named.


def test_amendment_and_timing_norm_of_another_act_go_to_generation() -> None:
    answer, assessment = scoped_evidence_response([(AMENDMENT, 0.9), (NORM, 0.8)], "", QUESTION)

    assert answer is None
    assert assessment is not None
    assert assessment.status == EvidenceStatus.COMPLETE
    assert [claim.label for claim in assessment.claims] == ["предмет изменения", "норма о периодичности"]
    assert extractive_timing_response([(AMENDMENT, 0.9), (NORM, 0.8)], QUESTION) is None


def test_only_the_amendment_names_the_missing_timing_norm() -> None:
    answer, assessment = scoped_evidence_response([(AMENDMENT, 0.9)], "", QUESTION)

    assert assessment is not None
    assert assessment.status == EvidenceStatus.PARTIAL
    assert answer is not None
    assert "Код товара по ТН ВЭД ЕАЭС" in answer
    assert "[1]" in answer
    assert "Не подтверждены: норма, устанавливающая периодичность передачи сведений." in answer
    assert "действие, категория товаров" not in answer


def test_only_the_timing_norm_names_the_missing_amendment() -> None:
    answer, assessment = scoped_evidence_response([(NORM, 0.8)], "", QUESTION)

    assert assessment is not None
    assert assessment.status == EvidenceStatus.PARTIAL
    assert answer is not None
    assert "в режиме реального времени" in answer
    assert "[2]" in answer
    assert "Не подтверждены: текст изменений, внесённых названным актом." in answer
