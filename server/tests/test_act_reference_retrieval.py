"""Named legal acts steer retrieval; sibling chunks cannot crowd out other provisions."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.documents import Document
import pytest

from domain.rag.act_references import (
    act_focused_queries,
    document_act_numbers,
    extract_act_numbers,
    referenced_act,
)
from domain.rag.ranking import prioritize_referenced_acts
from infrastructure.ml.evidence_profiles.profiles import GenericEvidenceProfile, evidence_profile_scope
from infrastructure.ml.rag.rag_reranking import rerank_documents
from infrastructure.ml.rag.retrieval_pipeline import retrieve_with_decomposition


LAW = "Закон Республики Беларусь от 28.12.2009 № 113-З (ред. от 16.03.2026) Об электронном документе.docx"
MNS_11 = "Постановление Министерства по налогам и сборам Республики Беларусь от 09.04.2021 N 11.docx"
MNS_13 = "Постановление МНС РБ от 19.04.2021 N 13 (ред. от 28.08.2025).docx"
AMENDMENT = "Постановление НАН Беларуси, Минфина РБ, МНС РБ 24 сентября 2024 г. N 11493026.docx"
COMPARISON = (
    "Как соотносятся требования к электронной цифровой подписи в Законе N 113-З "
    "и в постановлении МНС N 11 о представлении сведений EDI-провайдерами?"
)


def chunk(text: str, source: str, document_id: int, section: str | None = None) -> Document:
    metadata: dict[str, object] = {"document_id": document_id, "source": source}

    if section:
        metadata["section"] = section

    return Document(page_content=text, metadata=metadata)


def scored_reranker(scores: dict[str, float]) -> MagicMock:
    def predict(pairs: list[tuple[str, str]]) -> list[float]:
        return [next(score for body, score in scores.items() if text.endswith(body)) for _, text in pairs]

    return MagicMock(predict=predict)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("согласно постановлению от 24.09.2024 N 11/49/30/26", ("11493026",)),
        ("в Законе N 113-З и в постановлении МНС N 11", ("113-З", "11")),
        ("Закона № 113-з", ("113-З",)),
        ("постановлениям МНС N 11 и N 13, повторно N 11", ("11", "13")),
        ("пункт 11, приложение 4-1, п. 5", ()),
        ("что сказано в пункте № 3 Закона N 113-З", ("113-З",)),
        ("приложение N 2 к постановлению МНС N 11, ст. № 5, подпункт N 4", ("11",)),
    ],
)
def test_act_numbers_are_normalised_like_source_titles(text: str, expected: tuple[str, ...]) -> None:
    assert extract_act_numbers(text) == expected


def test_document_title_matches_whole_act_number_only() -> None:
    assert document_act_numbers({"source": f"/data/{AMENDMENT}"}) == {"11493026"}
    assert referenced_act({"source": LAW}, ("11", "113-З")) == "113-З"
    # "N 11" must not match acts numbered 113-З or 11493026.
    assert referenced_act({"source": LAW}, ("11",)) is None
    assert referenced_act({"source": AMENDMENT}, ("11",)) is None
    assert referenced_act({"doc_title": MNS_11.removesuffix(".docx")}, ("11",)) == "11"


def test_comparison_gets_one_focused_query_per_act() -> None:
    first, second = act_focused_queries(COMPARISON)
    assert "113-З" in first
    assert "МНС" not in first
    assert "EDI-провайдер" not in first
    assert "N 11 о представлении сведений EDI-провайдерами" in second
    assert "113-З" not in second
    assert all(
        query.startswith("Как соотносятся требования к электронной цифровой подписи")
        for query in (first, second)
    )


def test_provision_number_does_not_turn_a_question_into_a_comparison() -> None:
    assert act_focused_queries("Что сказано в пункте № 3 и в Законе N 113-З?") == []


def test_untyped_second_act_inherits_the_act_type() -> None:
    queries = act_focused_queries("Кто такой EDI-провайдер согласно постановлениям МНС N 11 и N 13?")
    assert queries == [
        "Кто такой EDI-провайдер: постановлениям МНС N 11",
        "Кто такой EDI-провайдер: постановлениям МНС N 13",
    ]


@pytest.mark.parametrize(
    "question",
    [
        "Как изменился формат поля 23 согласно постановлению от 24.09.2024 N 11/49/30/26?",
        "Какой срок хранения накладной?",
    ],
)
def test_single_or_no_act_needs_no_extra_queries(question: str) -> None:
    assert act_focused_queries(question) == []


def test_named_act_is_promoted_into_window_without_adding_evidence() -> None:
    other = [(chunk(f"other {i}", MNS_13, 13, f"point {i}"), 0.99 - i / 100) for i in range(4)]
    law = [(chunk(f"law {i}", LAW, 113, f"article {i}"), 0.5 - i / 100) for i in range(3)]
    items = [*other, *law]
    ranked = prioritize_referenced_acts(items, ("113-З",), top_n=4)
    window = ranked[:4]
    assert [doc.page_content for doc, _ in window] == ["law 0", "law 1", "other 0", "other 1"]
    assert sorted(id(item) for item in ranked) == sorted(id(item) for item in items)


def test_named_acts_share_the_window_fairly() -> None:
    law = [(chunk(f"law {i}", LAW, 113, f"article {i}"), 0.9 - i / 100) for i in range(4)]
    mns = [(chunk(f"mns {i}", MNS_11, 11, f"point {i}"), 0.5 - i / 100) for i in range(3)]
    ranked = prioritize_referenced_acts([*law, *mns], ("113-З", "11"), top_n=4)
    assert [doc.page_content for doc, _ in ranked[:4]] == ["law 0", "law 1", "mns 0", "mns 1"]


def test_guaranteed_share_of_a_named_act_is_never_taken() -> None:
    law = [(chunk(f"law {i}", LAW, 113, f"article {i}"), 0.9 - i / 100) for i in range(2)]
    mns = [(chunk(f"mns {i}", MNS_11, 11, f"point {i}"), 0.5 - i / 100) for i in range(2)]
    lone = [(chunk("mns13", MNS_13, 13, "point 1"), 0.1)]
    ranked = prioritize_referenced_acts([*law, *mns, *lone], ("113-З", "11", "13"), top_n=4)
    assert [doc.page_content for doc, _ in ranked] == ["law 0", "law 1", "mns 0", "mns 1", "mns13"]


def test_without_named_act_evidence_order_is_unchanged() -> None:
    items = [(chunk(f"other {i}", MNS_13, 13), 0.9 - i / 10) for i in range(3)]
    assert prioritize_referenced_acts(items, ("113-З",), top_n=2) == items
    assert prioritize_referenced_acts(items, (), top_n=2) == items


@pytest.mark.asyncio
async def test_rerank_prefers_act_named_in_question_over_similar_act() -> None:
    wrong = chunk("Сведения о товарах представляются в формате JSON.", MNS_13, 13, "point 4")
    right = chunk("Сведения передаются в режиме реального времени.", MNS_11, 11, "point 2 > subpoint_num 2.2")
    reranker = scored_reranker({wrong.page_content: 0.997, right.page_content: 0.96})
    ranked = await rerank_documents(
        "В каком режиме передаются сведения согласно постановлению МНС N 11?", [wrong, right], 2, reranker
    )
    assert [doc.metadata["document_id"] for doc, _ in ranked] == [11, 13]


@pytest.mark.asyncio
async def test_generic_profile_keeps_reranker_order_for_named_acts() -> None:
    wrong = chunk("Сведения о товарах представляются в формате JSON.", MNS_13, 13, "point 4")
    right = chunk("Сведения передаются в режиме реального времени.", MNS_11, 11, "point 2")
    reranker = scored_reranker({wrong.page_content: 0.997, right.page_content: 0.96})

    with evidence_profile_scope(GenericEvidenceProfile()):
        ranked = await rerank_documents(
            "В каком режиме передаются сведения согласно постановлению МНС N 11?", [wrong, right], 2, reranker
        )

    assert [doc.metadata["document_id"] for doc, _ in ranked] == [13, 11]


@pytest.mark.asyncio
async def test_comparison_keeps_both_acts_within_top_n() -> None:
    mns = [chunk(f"Подпись EDI-провайдера {i}.", MNS_11, 11, f"point {i}") for i in range(4)]
    law = chunk("Электронная цифровая подпись вырабатывается средством ЭЦП.", LAW, 113, "Статья 13")
    scores = {doc.page_content: 0.99 - i / 1000 for i, doc in enumerate(mns)}
    scores[law.page_content] = 0.4
    ranked = await rerank_documents(COMPARISON, [*mns, law], 4, scored_reranker(scores))
    assert law in [doc for doc, _ in ranked]
    assert len(ranked) == 4


@pytest.mark.asyncio
async def test_sibling_sentences_of_one_provision_cannot_crowd_out_another_provision() -> None:
    """Benchmark case: eight sentence chunks of subpoint 2.3 hid subpoint 2.2."""
    section = "point 2 > subpoint_num 2.3 > sentence"
    siblings = [chunk(f"2.3 предложение {i}.", MNS_11, 11, section) for i in range(8)]
    answer = chunk("2.2 в режиме реального времени.", MNS_11, 11, "point 2 > subpoint_num 2.2")
    scores = {doc.page_content: 0.997 - i / 1000 for i, doc in enumerate(siblings)}
    scores[answer.page_content] = 0.97
    ranked = await rerank_documents("В каком режиме?", [*siblings, answer], 4, scored_reranker(scores))
    assert answer in [doc for doc, _ in ranked]


@pytest.mark.asyncio
async def test_comparison_question_searches_each_act_under_the_same_filter() -> None:
    rag = MagicMock()
    rag.features.decomposition_enabled = False
    access_filter = object()
    state = SimpleNamespace(
        retrieval_filter=access_filter,
        query_domain="general",
        fetch_k=10,
        rag=rag,
        effective_dense_weight=0.7,
        effective_sparse_weight=0.3,
        full_corpus_searches=set(),
        visibility_conditions=[object()],
        user=SimpleNamespace(user_id=1),
        ctx=SimpleNamespace(user_group_ids=(5,)),
    )
    search = AsyncMock(return_value=[])

    with (
        patch("infrastructure.ml.rag.retrieval_pipeline.run_hybrid_search", search),
        patch(
            "infrastructure.ml.rag.retrieval_pipeline.policy_for",
            return_value=SimpleNamespace(fallback_to_full_corpus=False),
        ),
    ):
        _, sub_queries = await retrieve_with_decomposition(COMPARISON, state, MagicMock())  # type: ignore[arg-type]

    assert sub_queries[1:] == act_focused_queries(COMPARISON)
    assert [call.args[0] for call in search.await_args_list] == sub_queries
    assert all(call.args[2] is access_filter for call in search.await_args_list)
    assert all(call.kwargs["user_group_ids"] == [5] for call in search.await_args_list)


@pytest.mark.asyncio
async def test_generic_profile_adds_no_act_queries() -> None:
    rag = MagicMock()
    rag.features.decomposition_enabled = False
    state = SimpleNamespace(
        retrieval_filter=object(),
        query_domain="general",
        fetch_k=10,
        rag=rag,
        effective_dense_weight=0.7,
        effective_sparse_weight=0.3,
        full_corpus_searches=set(),
        visibility_conditions=[object()],
        user=SimpleNamespace(user_id=1),
        ctx=SimpleNamespace(user_group_ids=(5,)),
    )
    search = AsyncMock(return_value=[])

    with (
        evidence_profile_scope(GenericEvidenceProfile()),
        patch("infrastructure.ml.rag.retrieval_pipeline.run_hybrid_search", search),
        patch(
            "infrastructure.ml.rag.retrieval_pipeline.policy_for",
            return_value=SimpleNamespace(fallback_to_full_corpus=False),
        ),
    ):
        _, sub_queries = await retrieve_with_decomposition(COMPARISON, state, MagicMock())  # type: ignore[arg-type]

    assert sub_queries == [COMPARISON]
    assert search.await_count == 1
