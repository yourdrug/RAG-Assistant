"""Evidence on the question's subject survives reranking: classifier references (case 27)
and a short topic document read whole when every candidate was rejected (case 550)."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from langchain_core.documents import Document
import pytest

from domain.identity.roles import UserKind, UserRole
from domain.identity.user_context import UserContext
from domain.knowledge.chunk import ChunkRecord
from domain.knowledge.enums import DocumentVisibility
from domain.rag.topic_document import complete_indexes, names_subject, subject_stems
from infrastructure.domain_packs.belarus_law.evidence.applicability import (
    approved_classifier,
    classifier_reference,
)
from infrastructure.domain_packs.belarus_law.evidence.profile import BelarusEvidenceProfile
from infrastructure.ml.rag.topic_document import (
    MAX_TOPIC_DOCUMENT_CHUNKS,
    complete_topic_context,
    topic_document_fallback,
)


if TYPE_CHECKING:
    from domain.rag.ports import ChunkRetrievalRepository


CASE_27 = (
    "Какой орган утверждает классификатор дополнительной таможенной информации, на который есть ссылка "
    "из структуры сведений EDI-провайдеров и структуры ЭТТН?"
)
GTK_ENACTING = (
    "(в ред. постановления ГТК от 18.10.2019 N 44)\n"
    "классификатор дополнительной таможенной информации согласно приложению 4-1."
)
ETTN_ROW = (
    "| 42 | Дополнительный четырехзначный код по классификатору дополнительной таможенной информации an..4 "
    "| М | <ItemCustomCodeOther>4023</ItemCustomCodeOther> | Дополнительный код "
    "в соответствии с приложением 4-1 к постановлению Государственного таможенного комитета "
    "Республики Беларусь от 12 февраля 2016 г. N 5 |"
)
EDI_PROCESS = (
    "4. Сведения о товарах, указанных в ЭТТН, поступившей в систему электронного документооборота "
    "EDI-провайдера, представляются по структуре согласно приложению 1."
)
CASE_550 = "Какой лимит стоимости гостиницы в командировке?"
USER = UserContext(user_id=3, user_kind=UserKind.INTERNAL, user_role=UserRole.USER, group_ids=())


def test_classifier_approval_question_keeps_references_to_the_establishing_act() -> None:
    profile = BelarusEvidenceProfile()

    assert approved_classifier(CASE_27) == "дополнительной таможенной информации"
    assert profile.requested_provision(CASE_27, GTK_ENACTING)
    assert profile.requested_provision(CASE_27, ETTN_ROW)
    # The referring structure's own vocabulary is not a reference to the classifier.
    assert not profile.requested_provision(CASE_27, EDI_PROCESS)


def test_classifier_reference_needs_the_question_classifier_and_an_appendix() -> None:
    other = "классификатор видов транспорта согласно приложению 3."
    unreferenced = "Четырехзначный код по классификатору дополнительной таможенной информации."

    assert not classifier_reference("дополнительной таможенной информации", other)
    assert not classifier_reference("дополнительной таможенной информации", unreferenced)
    assert (
        approved_classifier("Что указывается в классификаторе дополнительной таможенной информации?") is None
    )


def test_subject_stems_skip_value_words_and_match_inflections() -> None:
    stems = subject_stems(CASE_550)

    assert "лимит" not in "".join(stems)
    assert names_subject("Порядок согласования командировок учебной компании", stems)
    assert not names_subject("Сведения об учебном офисе", stems)


def test_complete_indexes_require_contiguous_document_shorter_than_window() -> None:
    assert complete_indexes([0, 1, 2], 12)
    assert not complete_indexes([0, 2, 3], 12)
    assert complete_indexes(list(range(12)), 12)
    # get_neighbors(0, 12) reads indexes 0..12: the 13th chunk shows the document goes on.
    assert not complete_indexes(list(range(13)), 12)
    assert not complete_indexes([], 12)


def record(document_id: int, index: int, content: str, title: str) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=document_id * 100 + index,
        document_id=document_id,
        filename=f"{title}.txt",
        content=content,
        chunk_index=index,
        visibility=DocumentVisibility.INTERNAL_PUBLIC,
        doc_title=title,
    )


TRAVEL = [
    record(14, 0, "Порядок согласования командировок учебной компании «Маяк»", "travel-policy"),
    record(
        14,
        1,
        "# Планирование\nЗаявку подают за пять рабочих дней. Финансовый отдел утверждает бюджет.",
        "travel-policy",
    ),
    record(
        14,
        2,
        "# Изменение поездки\nИзменение стоимости более чем на 10 процентов требует согласования.",
        "travel-policy",
    ),
]
OFFICE = [
    record(
        15,
        0,
        "Сведения об учебном офисе «Маяка»\n# Посещение\nГостевой пропуск оформляет офис-менеджер.",
        "office",
    )
]


class NeighborRepository:
    def __init__(self, documents: dict[int, list[ChunkRecord]]) -> None:
        self.documents = documents
        self.calls: list[tuple[int, int, int, UserContext]] = []

    async def get_neighbors(
        self,
        document_id: int,
        center_index: int,
        window: int = 1,
        exclude_hashes: object = None,  # noqa: ARG002 - port signature
        *,
        user: UserContext,
    ) -> list[ChunkRecord]:
        self.calls.append((document_id, center_index, window, user))
        return [c for c in self.documents.get(document_id, []) if abs(c.chunk_index - center_index) <= window]


def outline_port(documents: dict[int, list[ChunkRecord]]) -> ChunkRetrievalRepository:
    return cast("ChunkRetrievalRepository", NeighborRepository(documents))


def candidate(document_id: int) -> Document:
    return Document(page_content="x", metadata={"document_id": document_id})


@pytest.mark.asyncio
async def test_topic_document_is_read_whole_with_a_completeness_note() -> None:
    fake = NeighborRepository({14: TRAVEL, 15: OFFICE})
    repository = cast("ChunkRetrievalRepository", fake)

    docs = await topic_document_fallback(CASE_550, [candidate(15), candidate(14)], repository, USER)

    assert [doc.metadata.get("chunk_index") for doc, _ in docs] == [None, 0, 1, 2]
    assert "приведён полностью: 3 фрагментов" in docs[0][0].page_content
    assert {doc.metadata["document_id"] for doc, _ in docs} == {14}
    assert complete_topic_context(docs)
    assert not complete_topic_context(docs[1:])
    assert all(call[3] is USER and call[2] == MAX_TOPIC_DOCUMENT_CHUNKS for call in fake.calls)


@pytest.mark.asyncio
async def test_long_unrelated_or_ambiguous_documents_are_not_claimed() -> None:
    long_travel = [
        record(14, i, "Порядок согласования командировок" if i == 0 else "текст", "travel") for i in range(20)
    ]
    second_travel = [record(16, 0, "Положение о командировках", "travel-2")]

    long_docs = await topic_document_fallback(
        CASE_550, [candidate(14)], outline_port({14: long_travel}), USER
    )
    unrelated = await topic_document_fallback(CASE_550, [candidate(15)], outline_port({15: OFFICE}), USER)
    ambiguous = await topic_document_fallback(
        CASE_550,
        [candidate(14), candidate(16)],
        outline_port({14: TRAVEL, 16: second_travel}),
        USER,
    )

    assert long_docs == []
    assert unrelated == []
    assert ambiguous == []
