"""Regressions for act identities, literal XML, partial answers and boundaries."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import docx
from langchain_core.documents import Document
import pytest

from domain.benchmark.evaluation import evaluate_evidence
from domain.knowledge.chunk import ChunkRecord
from infrastructure.metrics import metrics
from infrastructure.ml.evidence_profiles.belarus.grounded_response import amendment_field_response
from infrastructure.ml.evidence_profiles.belarus.queries import evidence_search_queries
from infrastructure.ml.ingestion.act_identity import act_heading_title
from infrastructure.ml.langchain_document_parser import LangchainDocumentParser, act_title_with_subject
from infrastructure.ml.rag import rag_postprocess
from infrastructure.ml.rag.context_rendering import build_header, render_selected_docs
from infrastructure.ml.rag.prompt_blocks import is_not_found_answer
from infrastructure.ml.rag.prompt_budget import estimate_text_tokens
from infrastructure.ml.rag.prompts import build_system_prompt
from infrastructure.ml.rag.rag_postprocess import mark_partial_answer


def test_docx_act_requisites_survive_filename_and_propagate_to_every_section(tmp_path: Path) -> None:
    path = tmp_path / "temporary.docx"
    word = docx.Document()
    word.add_paragraph("ПОСТАНОВЛЕНИЕ")
    word.add_paragraph("24 сентября 2024 г. N 11/49/30/26")
    word.add_paragraph("1. Внести изменения в постановление от 18 декабря 2023 г. N 9/75/35/26.")
    word.add_heading("Приложение", level=1)
    word.add_paragraph("Текст приложения.")
    word.save(str(path))
    docs = LangchainDocumentParser().parse(path, filename="Постановление N 11493026.docx")
    assert len(docs) > 1

    for doc in docs:
        assert "11/49/30/26" in doc.metadata["doc_title"]
        assert "9/75/35/26" not in doc.metadata["doc_title"]
        assert doc.metadata["doc_date"] == "2024-09-24"
        header = build_header(Document(page_content=doc.page_content, metadata=doc.metadata), 1)
        assert "11/49/30/26" in header
        assert "11493026" not in header


@pytest.mark.parametrize(
    "text",
    [
        "Постановление\n1. Внести изменения в постановление от 18.12.2023 N 9/75/35/26.",
        "Постановление\n(в ред. постановления от 24.09.2024 N 11/49/30/26)",
        "Постановление\n31.02.2024 N 1",
        "Приложение\n24.09.2024 N 11/49/30/26",
    ],
)
def test_referenced_acts_and_invalid_headings_are_not_document_identity(text: str) -> None:
    assert act_heading_title(text) is None


def test_prompt_fragment_metrics_normalize_legacy_angles_and_cell_breaks() -> None:
    literal = "<ItemCustomCodeOther>4023</ItemCustomCodeOther><br> Обязательное поле"
    legacy = literal.replace("<", "\u2039").replace(">", "\u203a")
    annotations = {"expected_fragments": [{"source": "act.docx", "text": literal}]}
    selected = [{"content": literal, "metadata": {"source": "act.docx"}}]
    metrics, _ = evaluate_evidence(annotations, selected, selected, legacy, "")
    assert metrics["context_fragment_recall"] == 1
    indexed = literal.replace("<br>", "\n")
    selected[0]["content"] = indexed
    metrics, _ = evaluate_evidence(annotations, selected, selected, indexed, "")
    assert metrics["context_fragment_recall"] == 1


@pytest.mark.parametrize("boundary", ["ровно", "не менее", "свыше", "превышающий"])
def test_boundary_question_gets_conditions_in_narrow_prompt(boundary: str) -> None:
    prompt = build_system_prompt(question=f"Какой код при сроке {boundary} 120 дней?")
    assert "<conditional_rules_expansion>" in prompt
    assert "Краткость не отменяет эти условия" in prompt


@pytest.mark.parametrize(
    "answer",
    [
        "Для перевозок применяется ТТН-1 [1]. Информация не найдена в документах.",
        "Информация не найдена в документах. Для перевозок применяется ТТН-1 [1].",
        "В предоставленных документах указаны ТТН-1 и ТН-2.",
    ],
)
def test_substantive_answer_is_not_a_refusal(answer: str) -> None:
    assert not is_not_found_answer(answer)


@pytest.mark.parametrize(
    "answer",
    [
        "Информация не найдена в документах [1].",
        "В предоставленных документах информация по этому вопросу не найдена.",
        "В документах нет информации по этому вопросу.",
        "К сожалению, не удалось найти информацию.",
    ],
)
def test_refusal_variants_and_cited_refusals_are_not_found(answer: str) -> None:
    assert is_not_found_answer(answer)
    assert mark_partial_answer(answer) == answer


def test_answer_metrics_classify_refusals_like_postprocessing() -> None:
    with patch.object(metrics, "RAG_NOT_FOUND_TOTAL") as not_found_total:
        metrics.record_rag_answer("narrow", "В предоставленных документах указаны ТТН-1 [1].", 1, 0.5)
        not_found_total.inc.assert_not_called()
        metrics.record_rag_answer("narrow", "Информация не найдена в документах [1].", 1, 0.5)
        not_found_total.inc.assert_called_once()


def test_mixed_answer_marks_partial_instead_of_dropping_sources() -> None:
    answer = mark_partial_answer("Применяется ТТН-1 [1].\n\nИнформация не найдена в документах.")
    assert "Ответ частичный" in answer
    assert "Применяется ТТН-1 [1]" in answer
    assert not is_not_found_answer(answer)
    assert is_not_found_answer(mark_partial_answer("Информация не найдена в документах."))


def test_status_amendment_renders_obligation_without_xml_dump() -> None:
    question = "Как изменился статус обязательности поля «Код» (позиция 57 таблицы 4.1)?"
    context = (
        "[1] Постановление\nв таблице 4.1 позицию\n"
        "| 57 | Код <br> an..4 | С | <Code>4023</Code> |\nзаменить позицией\n"
        "| 57 | Код <br> an..4 | М - для прослеживаемых, <br> С - для прочих | <Code>4023</Code> |"
    )
    answer, _ = amendment_field_response(context, question)
    assert "ДО изменения — С" in answer
    assert "М - для прослеживаемых" in answer
    assert "С - для прочих" in answer
    assert "[1]" in answer
    assert "<Code>" not in answer
    assert "<br>" not in answer


def test_classifier_authority_probe_does_not_include_referring_structure() -> None:
    question = (
        "Какой орган утверждает классификатор дополнительной таможенной информации, "
        "на который есть ссылка из структуры сведений EDI-провайдеров и структуры ЭТТН?"
    )
    queries = evidence_search_queries(question)
    assert queries == [question, "Утверждение: классификатор дополнительной таможенной информации"]


@pytest.mark.asyncio
async def test_neighbors_share_rendered_token_budget_and_skip_oversized_neighbor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = Document(page_content="Правило", metadata={"document_id": 1, "source": "act.docx"})
    small = ChunkRecord(2, 1, "act.docx", "Исключение", 2, content_hash="small")
    huge = ChunkRecord(1, 1, "act.docx", "Большой текст " * 100, 1, content_hash="huge")
    fetch = AsyncMock(return_value=[huge, small])
    monkeypatch.setattr(rag_postprocess, "fetch_context_neighbors", fetch)
    result = await rag_postprocess.enrich_with_neighbors(
        [(seed, 1.0)], enumerate_cases=True, chunk_search=SimpleNamespace(), max_context_tokens=100
    )
    assert [doc.page_content for doc, _ in result] == ["Правило", "Исключение"]
    assert estimate_text_tokens(render_selected_docs(result)[0]) <= 100


@pytest.mark.parametrize(
    "title,expected",
    [
        (
            "Об утверждении Правил перевозки",
            "Постановление от 12 марта 2020 г. № 5 «Об утверждении Правил перевозки»",
        ),
        ("ПОСТАНОВЛЕНИЕ", "Постановление от 12 марта 2020 г. № 5"),
        (None, "Постановление от 12 марта 2020 г. № 5"),
    ],
)
def test_act_requisites_keep_descriptive_title(title: str | None, expected: str) -> None:
    assert act_title_with_subject("Постановление от 12 марта 2020 г. № 5", title) == expected
