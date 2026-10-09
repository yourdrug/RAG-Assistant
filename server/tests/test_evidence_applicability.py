"""Scope failures must not become confident answers, including through generation."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from domain.value_objects.chat_context import ChatContext
from domain.value_objects.llm_provider import Breadth
from domain.value_objects.roles import UserKind, UserRole
from domain.value_objects.evidence_assessment import EvidenceStatus
from infrastructure.ml.rag.grounded_response import (
    extractive_procedure_response,
    extractive_table_comparison_response,
    extractive_timing_response,
    missing_field_response,
    scoped_evidence_response,
)
from infrastructure.ml.rag.rag_config import build_rag_settings
from infrastructure.ml.rag.rag_steps import step_build_context, step_generate
from infrastructure.ml.rag_pipeline import RagPipelineState

PROBES = json.loads((Path(__file__).parent / 'fixtures/evidence_scope_probes.json').read_text())['cases']
FAILURES = {
    'timing-unrelated-action',
    'timing-missing-category',
    'procedure-duplicate-one-stage',
    'procedure-other-act',
    'procedure-other-version',
    'procedure-only-shipping',
    'table-wrong-explicit-table',
}


@pytest.mark.parametrize('case', [c for c in PROBES if c['id'] in FAILURES], ids=lambda c: c['id'])
def test_extractors_reject_unproven_applicability(case):
    docs = [(Document(**item), 0.95) for item in case.get('docs', [])]
    if case['kind'] == 'procedure':
        assert extractive_procedure_response(docs, case['question']) is None
    elif case['kind'] == 'timing':
        assert extractive_timing_response(docs, case['question']) is None
    else:
        assert missing_field_response(case['context'], case['question'])
        assert extractive_table_comparison_response(case['context'], case['question']) is None


@pytest.mark.asyncio
@pytest.mark.parametrize('case', PROBES, ids=lambda c: c['id'])
async def test_incomplete_evidence_cannot_escape_to_a_guessing_llm(case):
    docs = [(Document(**item), 0.95) for item in case.get('docs', [])]
    if not docs:
        docs = [(Document(page_content=case['context'], metadata={'source': 'table.docx'}), 0.95)]
    ctx = ChatContext(1, UserKind.INTERNAL, UserRole.ADMIN)
    state = RagPipelineState(
        rag=build_rag_settings(),
        t_pipeline_start=0,
        question=case['question'],
        ctx=ctx,
        user=ctx.to_user_context(),
        access_filter=None,
        retrieval_filter=None,
        breadth=Breadth.NARROW,
        docs=docs,
    )
    messages, grouped = step_build_context(state, None)
    # No generation client: unsafe fall-through fails at the real boundary.
    events = [event async for event in step_generate(state, SimpleNamespace(), messages, grouped)]
    answer = ''.join(event.text for event in events)
    assert answer and state.evidence_assessment is not None
    if case.get('expect_decline'):
        assert state.evidence_assessment.status != EvidenceStatus.COMPLETE
    if case['id'] in FAILURES:
        assert any(word in answer.casefold() for word in ('не подтвержд', 'нельзя подтверд', 'конфликт'))
        assert 'Архивные отчеты' not in answer


def test_table_with_no_identity_does_not_confirm_field_value():
    case = next(c for c in PROBES if c['id'] == 'table-positive')
    assert missing_field_response(case['context'], case['question'])
    assert extractive_table_comparison_response(case['context'], case['question']) is None


def test_correct_table_identity_preserves_positive_comparison():
    case = next(c for c in PROBES if c['id'] == 'table-positive')
    context = case['context'].replace('Consolidated\n', 'Consolidated\nТаблица 7.2\n')
    answer = extractive_table_comparison_response(context, case['question'])
    assert answer and 'an..8. [1]' in answer and 'an..8 → an..12. [2]' in answer


def test_missing_actor_does_not_confirm_timing_for_requested_provider():
    question = 'В каком режиме EDI-провайдер передаёт сведения о товарах, включённых в перечни, по N 71?'
    doc = Document(
        page_content='Сведения о товарах, включенных в перечни, передаются архивариусом ежедневно.',
        metadata={
            'source': 'Постановление N 71.docx',
            'document_id': 71,
            'act_version_id': 1,
            'citation_id': 1,
        },
    )
    assert extractive_timing_response([(doc, 0.95)], question) is None


def test_act_number_inside_unrelated_filename_is_not_identity():
    case = next(c for c in PROBES if c['id'] == 'timing-positive')
    doc = Document(**case['docs'][0])
    doc.metadata['doc_title'] = 'Постановление N 72'
    assert extractive_timing_response([(doc, 0.95)], case['question']) is None


def test_disconnected_amendment_values_cannot_fall_through_to_generation():
    case = next(c for c in PROBES if c['id'] == 'amendment-missing-operation')
    context = case['context'].replace('Consolidated\n', 'Consolidated\nТаблица 7.2\n')
    answer, assessment = scoped_evidence_response([], context, case['question'].replace('формат', 'статус'))
    assert answer and assessment and 'не подтвержден' in answer.casefold()
    assert 'ПОСЛЕ изменения' not in answer


def test_complete_amendment_is_quoted_without_free_generation():
    case = next(c for c in PROBES if c['id'] == 'amendment-positive')
    context = case['context'].replace('Consolidated\n', 'Consolidated\nТаблица 7.2\n')
    answer, assessment = scoped_evidence_response([], context, case['question'].replace('формат', 'статус'))
    assert answer and assessment and 'ДО изменения' in answer and 'ПОСЛЕ изменения' in answer


@pytest.mark.parametrize('recipient', ['МНС', 'Министерство по налогам и сборам'])
def test_wrong_recipient_does_not_confirm_transmission_timing(recipient):
    question = (
        f'В каком режиме EDI-провайдер передает в {recipient} '
        'сведения о товарах, включенных в перечни, по N 71?'
    )
    doc = Document(
        page_content='EDI-провайдер передает сведения о товарах, включенных в перечни, в архив ежедневно.',
        metadata={'source': 'Постановление N 71.docx', 'citation_id': 1, 'document_id': 71},
    )
    assert extractive_timing_response([(doc, 0.95)], question) is None


def test_wrong_act_table_value_cannot_fall_through_or_be_quoted():
    question = 'Формат поля «Код» (позиция 18 таблицы 7.2 BLRDLN) постановления N 58?'
    context = '[1] Постановление N 99\nТаблица 7.2 BLRDLN\n| 18 | Код an..8 | М |'
    answer, assessment = scoped_evidence_response([], context, question)
    assert answer and assessment and 'не подтвержден' in answer.casefold()
    assert 'an..8' not in answer


def test_russian_message_name_does_not_accept_other_message_type():
    question = 'Формат поля «Код» (позиция 18 таблицы 7.2 сообщения ЭТН)?'
    context = '[1] Источник\nТаблица 7.2 BLRWBL\n| 18 | Код an..8 | М |'
    assert missing_field_response(context, question)


def test_field_name_must_match_whole_name():
    question = 'Формат поля «Код» (позиция 18 таблицы 7.2 BLRDLN)?'
    context = '[1] Источник\nТаблица 7.2 BLRDLN\n| 18 | Код получателя an..8 | М |'
    assert missing_field_response(context, question)


def test_mixed_table_versions_cannot_confirm_value():
    question = 'Формат поля «Код» (позиция 18 таблицы 7.2 BLRDLN) постановления N 58?'
    text = 'Таблица 7.2 BLRDLN\n| 18 | Код an..8 | М |'
    docs = [
        (
            Document(
                page_content=text,
                metadata={
                    'source': 'Постановление N 58.docx',
                    'document_id': 58,
                    'act_version_id': version,
                    'citation_id': version,
                },
            ),
            0.95,
        )
        for version in (1, 2)
    ]
    context = f'[1] Постановление N 58\n{text}\n\n---\n\n[2] Постановление N 58\n{text}'
    answer, _ = scoped_evidence_response(docs, context, question)
    assert answer and 'an..8' not in answer
