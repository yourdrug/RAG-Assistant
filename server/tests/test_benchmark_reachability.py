"""Fragment reachability: a labelled fragment must lie whole inside one chunk of the space."""

from __future__ import annotations

from typing import Any

import pytest

from application.benchmark.quality import BenchmarkQualityService
from domain.benchmark.question import BenchmarkQuestion
from domain.benchmark.reachability import UnreachableReason, fragment_reachability
from domain.knowledge.chunk import ChunkInsert
from domain.knowledge.repositories.chunk import ChunkInsertOptions
from domain.knowledge.space import KnowledgeSpace
from domain.shared.exceptions import ValidationError
from infrastructure.benchmark.chunk_documents import UowBenchmarkChunkDocuments
from tests.fakes import FakeUnitOfWorkFactory


def question(question_id: int, *fragments: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": question_id,
        "question": f"q{question_id}",
        "annotations": {"expected_fragments": list(fragments)} if fragments else None,
    }


def chunk(source: str, content: str, **metadata: Any) -> dict[str, Any]:
    return {"content": content, "metadata": {"source": source, **metadata}}


def test_fragments_are_reachable_only_whole_inside_one_chunk() -> None:
    documents = [
        chunk("dir/Закон (1).docx", "Ставка налога — 20 процентов."),
        chunk("Закон.docx", "Срок уплаты — до 22 числа."),
    ]
    report = fragment_reachability(
        [
            # Directories, copy suffixes, case and spacing do not matter.
            question(1, {"source": "закон.docx", "text": "ставка  налога — 20"}),
            # Split across two chunks: no single chunk holds it.
            question(2, {"source": "Закон.docx", "text": "20 процентов. Срок уплаты"}),
            question(3, {"source": "Другой.docx", "text": "ставка"}),
            question(
                4,
                {
                    "alternatives": [
                        {"source": "Другой.docx", "text": "ставка"},
                        {"source": "Закон.docx", "text": "до 22 числа"},
                    ]
                },
            ),
            question(5),
        ],
        documents,
    )

    assert (report.questions, report.fragments, report.reachable) == (4, 4, 2)
    assert report.share == 0.5
    assert [(item.question_id, item.reason) for item in report.unreachable] == [
        (2, UnreachableReason.NOT_IN_CHUNK),
        (3, UnreachableReason.NO_DOCUMENT),
    ]


def test_section_and_document_constraints_apply() -> None:
    documents = [chunk("a.docx", "ставка 20", section="Статья 1", document_id=7)]

    report = fragment_reachability(
        [
            question(1, {"source": "a.docx", "text": "ставка", "section": "Статья 1"}),
            question(2, {"source": "a.docx", "text": "ставка", "section": "Статья 2"}),
            question(3, {"source": "a.docx", "text": "ставка", "document_id": 8}),
        ],
        documents,
    )

    assert report.reachable == 1
    assert {item.question_id for item in report.unreachable} == {2, 3}


def test_empty_dataset_has_no_share() -> None:
    report = fragment_reachability([question(1)], [])

    assert (report.questions, report.fragments, report.share) == (0, 0, None)


@pytest.mark.asyncio
async def test_service_checks_chunks_of_the_dataset_pack_space() -> None:
    factory = FakeUnitOfWorkFactory()
    uow = factory._uow
    law = await uow.knowledge_spaces.create(KnowledgeSpace(name="Право", pack_key="belarus_law"))
    other = await uow.knowledge_spaces.create(KnowledgeSpace(name="Поддержка", pack_key="support"))
    uow.fake_benchmark_questions.created.extend(
        [
            BenchmarkQuestion(
                id=1,
                question="ставка?",
                dataset="belarus_law-main",
                annotations={"expected_fragments": [{"source": "law.docx", "text": "ставка 20"}]},
            ),
            BenchmarkQuestion(
                id=2,
                question="срок?",
                dataset="belarus_law-main",
                annotations={"expected_fragments": [{"source": "faq.docx", "text": "срок"}]},
            ),
            BenchmarkQuestion(id=3, question="inactive", dataset="belarus_law-main", is_active=False),
        ]
    )
    await uow.chunks.bulk_insert(
        10,
        "law.docx",
        "internal_public",
        [chunk_insert("Ставка 20 процентов")],
        ChunkInsertOptions(space_id=law.id),
    )
    # The same file in another space is not part of this corpus.
    await uow.chunks.bulk_insert(
        11, "faq.docx", "internal_public", [chunk_insert("срок")], ChunkInsertOptions(space_id=other.id)
    )
    service = BenchmarkQualityService(factory, UowBenchmarkChunkDocuments(factory))

    space, report = await service.fragment_reachability("belarus_law-main")

    assert space.id == law.id
    assert (report.fragments, report.reachable) == (2, 1)
    assert report.unreachable[0].reason is UnreachableReason.NO_DOCUMENT

    # An explicit space replaces the pack's space: only its documents count.
    _, explicit = await service.fragment_reachability("belarus_law-main", other.id)
    assert explicit.reachable == 1
    assert [(item.question_id, item.reason) for item in explicit.unreachable] == [
        (1, UnreachableReason.NO_DOCUMENT)
    ]

    await uow.knowledge_spaces.create(KnowledgeSpace(name="Право 2", pack_key="belarus_law"))

    with pytest.raises(ValidationError, match="choose space_id"):
        await service.fragment_reachability("belarus_law-main")


def chunk_insert(content: str) -> ChunkInsert:
    return ChunkInsert(content=content)
