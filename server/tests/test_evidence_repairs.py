"""Source links, missing evidence and negation survive production assembly."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from langchain_core.documents import Document
import pytest

from application.rag.stream_events import TextChunk
from domain.identity.roles import UserKind, UserRole
from domain.identity.user_context import UserContext
from domain.knowledge.access_control import VisibilityCondition
from domain.knowledge.chunk import ChunkRecord
from domain.knowledge.enums import DocumentVisibility, PageContentType
from domain.rag.breadth import Breadth
from domain.rag.context import RagContext
from infrastructure.ml.evidence_profiles.belarus.field_evidence import augment_field_candidates
from infrastructure.ml.evidence_profiles.belarus.focus import evidence_reading_map
from infrastructure.ml.evidence_profiles.belarus.grounded_response import (
    extractive_procedure_response,
    extractive_table_comparison_response,
    extractive_timing_response,
    missing_field_response,
)
from infrastructure.ml.evidence_profiles.belarus.linked_context import enrich_linked_context
from infrastructure.ml.evidence_profiles.belarus.predicates import contradicts_requested_scope
from infrastructure.ml.evidence_profiles.belarus.queries import evidence_search_queries
from infrastructure.ml.evidence_profiles.belarus.reading import evidence_reading_rules
from infrastructure.ml.rag import rag_steps
from infrastructure.ml.rag.chunk_metadata import chunk_result_metadata
from infrastructure.ml.rag.evidence_focus import format_generation_messages
from infrastructure.ml.rag.rag_config import build_rag_settings
from infrastructure.ml.rag.rag_formatting import format_docs_with_selection
from infrastructure.ml.rag.rag_prompts import build_prompt
from infrastructure.ml.rag.rag_reranking import filter_scope_candidates
from infrastructure.ml.rag.retrieval_pipeline import retrieve_with_decomposition
from infrastructure.ml.rag.retrieval_work import RetrievalWork
from tests.rag_stage_fixtures import check_cache as step_check_cache, generate as step_generate, stage_fixture
from tests.test_rag_pipeline import make_rag


FIELD = "Дополнительный четырёхзначный код по классификатору дополнительной таможенной информации"
AMENDMENT_QUESTION = f"Как изменился статус поля «{FIELD}» (позиция 57 таблицы 4.1)?"
PROCEDURE_QUESTION = "Опишите порядок: передача товара на хранение и последующая отгрузка."
USER = RagContext(
    user=UserContext(user_id=1, user_kind=UserKind.INTERNAL, user_role=UserRole.ADMIN)
).to_user_context()


def chunk(
    index: int,
    text: str,
    *,
    table: bool = False,
    section: str | None = None,
    document_id: int = 27,
    version: int = 2,
) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=index,
        document_id=document_id,
        filename="source.docx",
        content=text,
        chunk_index=index,
        act_version_id=version,
        section=section,
        content_type=PageContentType.TABLE.value if table else None,
    )


def hit(value: ChunkRecord) -> tuple[Document, float]:
    return Document(page_content=value.content, metadata=chunk_result_metadata(value)), 0.95


def amendment_chunks() -> list[ChunkRecord]:
    return [
        chunk(10, "1. Внести изменения:\nв таблице 4.1 - структура сообщения позицию", section="point 1"),
        chunk(11, f'| "57 | {FIELD} an..4 | С |', table=True),
        chunk(12, "заменить позицией"),
        chunk(13, f'| "57 | {FIELD} an..4 | М - для прослеживаемых, С - для прочих |', table=True),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", [1, 3])
async def test_legacy_amendment_recovers_atomic_old_operation_new(seed: int) -> None:
    chunks = amendment_chunks()
    search = SimpleNamespace(get_neighbors=AsyncMock(return_value=list(reversed(chunks))))
    original = hit(chunks[seed])
    docs = await enrich_linked_context([original], AMENDMENT_QUESTION, search, USER)
    search.get_neighbors.assert_awaited_once_with(27, chunks[seed].chunk_index, window=3, user=USER)
    assert "parent_units" not in original[0].metadata  # no shared/cache mutation
    context, selected = format_docs_with_selection(docs, 10000, context_counter=len)
    assert selected
    assert "заменить позицией" in context
    mapped = "\n".join(evidence_reading_map(context, AMENDMENT_QUESTION))
    assert mapped.index("ДО изменения") < mapped.index("| С |") < mapped.index("ПОСЛЕ изменения")
    assert "М - для прослеживаемых, С - для прочих" in mapped
    messages = format_generation_messages(
        build_prompt(question=AMENDMENT_QUESTION), context=context, history=[], question=AMENDMENT_QUESTION
    )
    assert "ДО изменения" in messages[-2].content
    assert "ПОСЛЕ изменения" in messages[-2].content
    context, selected = format_docs_with_selection(docs, len(context) - 1, context_counter=len)
    assert not selected
    assert not context


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["gap", "other_document", "other_version", "missing_operation"])
async def test_amendment_cannot_link_unrelated_or_missing_source_chunks(mutation: str) -> None:
    chunks = amendment_chunks()

    if mutation == "gap":
        chunks[0].chunk_index = 9
    elif mutation == "other_document":
        chunks[-1].document_id = 99
    elif mutation == "other_version":
        chunks[-1].act_version_id = 3
    else:
        chunks[2].content = "пояснение"

    search = SimpleNamespace(get_neighbors=AsyncMock(return_value=chunks))
    docs = await enrich_linked_context([hit(chunks[1])], AMENDMENT_QUESTION, search, USER)
    assert "parent_units" not in docs[0][0].metadata


@pytest.mark.asyncio
async def test_shipping_requisites_are_atomic_with_shipping_condition_only() -> None:
    storing = chunk(
        145, "При передаче на хранение составляется ТТН-1. Получатель — оператор.", section="point 3"
    )
    shipping = chunk(
        147, "При отгрузке по письменному указанию составляется ТТН-1 без стоимости.", section="point 3"
    )
    fields = chunk(
        148, 'В накладной ТТН-1 в строке "Грузоотправитель" указывается оператор.', section="point 3"
    )
    search = SimpleNamespace(get_neighbors=AsyncMock(return_value=[fields, storing, shipping]))
    docs = await enrich_linked_context([hit(storing), hit(fields)], PROCEDURE_QUESTION, search, USER)
    assert "parent_units" not in docs[0][0].metadata
    scope = docs[1][0].metadata["parent_units"][0]["content"]
    assert scope == shipping.content + "\n" + fields.content
    assert "хранение" not in scope


@pytest.mark.asyncio
async def test_procedure_never_joins_across_provisions() -> None:
    shipping = chunk(147, "При отгрузке составляется ТТН-1.", section="point 3")
    fields = chunk(
        148, 'В накладной ТТН-1 в строке "Грузоотправитель" указывается оператор.', section="point 4"
    )
    search = SimpleNamespace(get_neighbors=AsyncMock(return_value=[shipping, fields]))
    docs = await enrich_linked_context([hit(fields)], PROCEDURE_QUESTION, search, USER)
    assert "parent_units" not in docs[0][0].metadata


def test_missing_target_row_is_explicit_even_with_matching_name_or_number_elsewhere() -> None:
    snapshot = json.loads(
        (Path(__file__).parent / "fixtures/generation_completeness_reading.json").read_text()
    )
    case = next(case for case in snapshot["cases"] if case["id"] == 140)
    mapped = "\n".join(evidence_reading_map(case["context"], case["question"]))
    assert "строка 18 «УНП грузоотправителя»" in mapped
    assert "значение не подтверждено" in mapped
    assert "строка 23" in mapped  # A bare row without a table identity also remains unconfirmed.
    with_row = (
        case["context"]
        + "\n\n---\n\nТаблица 4.2, BLRDLN\n| 18 | УНП грузоотправителя \u2039br\u203a an..15 | 2:М |"
    )
    assert not any("строка 18" in item for item in evidence_reading_map(with_row, case["question"]))


@pytest.mark.asyncio
async def test_supplemental_retrieval_preserves_acl_temporal_filter_and_original_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    question = (
        "В каком режиме передают сведения о товарах, включённых в перечни, согласно постановлению N 11?"
    )
    probes = evidence_search_queries(question)
    assert len(probes) == 2
    assert probes[0] == question
    assert "режиме" in probes[1]
    assert "N 11" in probes[1]
    search = AsyncMock(side_effect=[[Document(page_content="не включенных")], [Document(page_content="2.2")]])
    monkeypatch.setattr("infrastructure.ml.rag.retrieval_pipeline.run_retrieval", search)
    acl = object()
    conditions = [VisibilityCondition(visibility=DocumentVisibility.INTERNAL_PUBLIC)]
    docs, queries = await retrieve_with_decomposition(
        question,
        RetrievalWork(
            fetch_k=30,
            retrieval_filter=acl,
            rag=make_rag(),
            breadth=Breadth.NARROW,
            query_domain="legal",
            effective_dense_weight=0.5,
            effective_sparse_weight=0.5,
            visibility_conditions=conditions,
            t_pipeline_start=0,
            question=question,
            access_filter=acl,
            ctx=RagContext(
                user=UserContext(
                    user_id=1, user_kind=UserKind.INTERNAL, group_ids=(7,), user_role=UserRole.USER
                )
            ),
            user=UserContext(user_id=1, user_kind=UserKind.INTERNAL, user_role=UserRole.USER),
        ),
        MagicMock(),
    )
    assert queries == probes
    assert [doc.page_content for doc in docs] == ["не включенных", "2.2"]

    for call in search.await_args_list:
        assert call.args[1].retrieval_filter is acl
        assert call.args[1].visibility_conditions is conditions
        assert call.args[1].ctx.user_group_ids == (7,)

    rules = "\n".join(evidence_reading_rules(question))
    assert "НЕ включённые" in rules
    assert "нужного подпункта нет" in rules


def test_table_comparison_probes_both_fields_without_answer_values() -> None:
    question = (
        "Изменился ли формат поля «УНП грузоотправителя» (позиция 18 таблицы 4.2), "
        "как поля «УНП грузополучателя» (позиция 23)?"
    )
    queries = evidence_search_queries(question)
    assert queries == [
        question,
        "Таблица 4.2, поле 18 «УНП грузоотправителя»",
        "Таблица 4.2, поле 23 «УНП грузополучателя»",
    ]
    assert evidence_search_queries("Где скачать приложение?") == ["Где скачать приложение?"]


@pytest.mark.asyncio
async def test_missing_field_lookup_uses_authorized_document_and_requested_date() -> None:
    ctx = RagContext(user=UserContext(user_id=1, user_kind=UserKind.INTERNAL, user_role=UserRole.ADMIN))
    question = "Формат поля «УНП грузоотправителя» (позиция 18 таблицы 4.2)?"
    seed = hit(chunk(616, "| 23 | УНП грузополучателя an..19 |", table=True))[0]
    valid = chunk(614, "| 18 | УНП грузоотправителя an..15 |", table=True)
    wrong_document = chunk(614, "| 18 | УНП грузоотправителя an..99 |", table=True, document_id=99)
    wrong_row = chunk(613, "| 15 | УНП грузоотправителя an..19 |", table=True)
    search = SimpleNamespace(search_substring=AsyncMock(return_value=[valid, wrong_document, wrong_row]))
    docs = await augment_field_candidates(question, [seed], search, ctx)
    assert len(docs) == 2
    assert docs[-1].page_content == valid.content
    call = search.search_substring.await_args.kwargs
    assert call["document_id"] == 27
    assert call["as_of_date"] == ctx.as_of_date
    assert call["user"] == ctx.to_user_context()


@pytest.mark.asyncio
async def test_missing_field_lookup_does_not_invent_evidence_on_empty_authorized_results() -> None:
    ctx = RagContext(user=UserContext(user_id=1, user_kind=UserKind.INTERNAL, user_role=UserRole.ADMIN))
    question = "Формат поля «УНП грузоотправителя» (позиция 18 таблицы 4.2)?"
    seed = hit(chunk(616, "| 23 | УНП грузополучателя an..19 |", table=True))[0]
    search = SimpleNamespace(search_substring=AsyncMock(return_value=[]))
    assert await augment_field_candidates(question, [seed], search, ctx) == [seed]
    assert "не подтверждено" in "\n".join(evidence_reading_map(seed.page_content, question))


@pytest.mark.parametrize("negative", [False, True])
def test_category_negation_filters_only_explicitly_opposite_scope(negative: bool) -> None:
    question = "Режим для товаров, " + ("не " if negative else "") + "включенных в перечни?"
    included = Document(page_content="Товары, включенные в перечни — категория А.")
    excluded = Document(page_content="Товары, не включенные в перечни — категория Б.")
    reference = Document(page_content="2.2. сведения, указанные в пункте 1.")
    expected = excluded if negative else included
    assert filter_scope_candidates([included, excluded, reference], question) == [expected, reference]
    both = "Сравните включенные в перечни и не включенные в перечни товары."
    assert not contradicts_requested_scope(both, excluded.page_content)


@pytest.mark.asyncio
async def test_field_lookup_rejects_same_number_and_name_from_wrong_message_table() -> None:
    ctx = RagContext(user=UserContext(user_id=1, user_kind=UserKind.INTERNAL, user_role=UserRole.ADMIN))
    question = "Формат поля «УНП грузоотправителя» (позиция 18 таблицы 4.2, BLRDLN)?"
    wrong = chunk(430, "Теги ЭТТН:\n| 18 | УНП грузоотправителя an..15 |", table=True)
    right = chunk(614, "Таблица 4.2\nТеги ЭТН:\n| 18 | УНП грузоотправителя an..15 |", table=True)
    search = SimpleNamespace(search_substring=AsyncMock(return_value=[wrong, right]))
    assert missing_field_response(wrong.content, question)
    docs = await augment_field_candidates(question, [hit(wrong)[0]], search, ctx)
    assert len(docs) == 1
    assert docs[0].page_content == right.content
    assert missing_field_response(right.content, question) is None


def test_extractive_procedure_keeps_complete_stage_links_in_source_order() -> None:
    question = "Опишите порядок взаимодействия с логистическим оператором согласно Инструкции."
    storage = chunk(145, "При передаче логистическому оператору товара на хранение получатель — оператор.")
    shipping = chunk(147, "При отгрузке логистическим оператором ТТН-1 стоимость может не указываться.")
    requisites = 'В накладной ТТН-1 в строке "Грузоотправитель" указывается оператор.'
    billing = chunk(
        160, "При реализации с участием логистических операторов после отгрузки ТН-2 содержит стоимость."
    )
    docs = [hit(value) for value in [billing, shipping, storage]]

    for doc, _ in docs:
        doc.metadata["citation_id"] = 1

    docs[1][0].metadata["parent_units"] = [{"content": shipping.content + "\n" + requisites}]
    answer = extractive_procedure_response(docs, question)
    assert answer is not None
    assert answer.index(storage.content) < answer.index(shipping.content) < answer.index(billing.content)
    assert answer is not None
    assert answer.count(requisites) == 1
    assert answer is not None
    assert answer.count("[1]") == 3


@pytest.mark.asyncio
async def test_missing_row_response_cannot_be_overridden_by_model_and_cache_changes_with_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = RagContext(user=UserContext(user_id=1, user_kind=UserKind.INTERNAL, user_role=UserRole.ADMIN))
    question = "Формат поля «УНП грузоотправителя» (позиция 18 таблицы 4.2, BLRDLN)?"
    state = stage_fixture(
        rag=build_rag_settings(),
        t_pipeline_start=0,
        question=question,
        ctx=ctx,
        user=ctx.to_user_context(),
        access_filter=None,
        retrieval_filter=None,
        breadth=Breadth.NARROW,
        query_for_search=question,
        grounded_answer=missing_field_response("", question),
    )
    events = [event async for event in step_generate(state, object(), [], [])]
    assert state.output_sanitized
    assert isinstance(events[0], TextChunk)
    assert "Нельзя подтвердить" in events[0].text
    assert "an..15" not in state.full_answer
    state.rag = replace(state.rag, features=replace(state.rag.features, cache_enabled=True))
    monkeypatch.setattr(rag_steps, "get_corpus_revision", AsyncMock(return_value="corpus"))
    monkeypatch.setattr(rag_steps, "check_cache", AsyncMock(return_value=None))
    await step_check_cache(state)
    original = state.q_hash
    monkeypatch.setattr(
        "infrastructure.ml.evidence_profiles.belarus.focus.EVIDENCE_POLICY_VERSION", "previous-policy"
    )
    await step_check_cache(state)
    assert state.q_hash != original


def test_timing_response_requires_unique_norm_in_requested_act_and_correct_category() -> None:
    question = "В каком режиме передают сведения о товарах, включённых в перечни, по постановлению N 11?"
    right = Document(
        page_content=(
            "Сведения о товарах, включенных в перечни, передаются провайдером "
            "в режиме реального времени в момент поступления;"
        ),
        metadata={"source": "Постановление N 11.docx", "citation_id": 1},
    )
    wrong = Document(
        page_content="Не включенные в перечни товары передаются раз в сутки;",
        metadata={"source": "Постановление N 11.docx", "citation_id": 1},
    )
    other = Document(
        page_content="Сведения о товарах, включенных в перечни, передаются в режиме ежедневной передачи;",
        metadata={"source": "Постановление N 110.docx", "citation_id": 2},
    )
    assert (
        extractive_timing_response([(wrong, 0.99), (right, 0.95), (other, 0.9)], question)
        == right.page_content + " [1]"
    )
    assert extractive_timing_response([(wrong, 0.99), (other, 0.9)], question) is None
    other.metadata["source"] = right.metadata["source"]
    assert extractive_timing_response([(right, 0.95), (other, 0.9)], question) is None


def test_table_comparison_cites_target_row_and_replacement_separately_without_inventing_history() -> None:
    question = (
        "Изменился ли формат поля «УНП грузоотправителя» (позиция 18 таблицы 4.2, BLRDLN), "
        "как поля «УНП грузополучателя» (позиция 23)?"
    )
    context = (
        "[1] Consolidated\nТаблица 4.2\nТеги ЭТН:\n| 18 | УНП грузоотправителя an..15 | 2:М |\n\n---\n\n"
        "[2] Amendment\nв таблице 4.2 - структура сообщения BLRDLN позицию\n"
        '| "23 | УНП грузополучателя an..15 | 3:М |\nзаменить позицией\n'
        '| "23 | УНП грузополучателя an..19 | 3:М |'
    )
    answer = extractive_table_comparison_response(context, question)
    assert answer is not None
    assert "в целевой строке указан формат an..15. [1]" in answer
    assert answer is not None
    assert "формат изменён an..15 → an..19. [2]" in answer
    assert "до изменения" not in answer.casefold()
    assert extractive_table_comparison_response("", question) is None
