"""Benchmark seed: question snapshot reconciliation, its file format and the packs' seeds."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from application.benchmark.dto import BenchmarkQuestionCreateDTO
from application.benchmark.services import BenchmarkQuestionService
from application.spaces.readiness import CorpusReadiness, corpus_readiness
from composition.cli.commands.seed import manual_copy_conflict
from composition.domain_packs import PACK_REGISTRATIONS, selected_pack
from domain.benchmark.question import BenchmarkQuestion
from domain.benchmark.question_snapshot import plan_question_sync
from domain.knowledge.document import Document
from domain.knowledge.enums import DocumentStatus, DocumentVisibility
from domain.shared.exceptions import ValidationError
from infrastructure.benchmark.seed import load_pack_seed, read_questions, write_questions
from infrastructure.domain_profile.packs import NEUTRAL_PACK
from tests.fakes import FakeBenchmarkQuestionRepository, FakeUnitOfWork, FakeUnitOfWorkFactory


if TYPE_CHECKING:
    from pathlib import Path

    from infrastructure.domain_profile.packs import DomainPack


PUBLIC = DocumentVisibility.INTERNAL_PUBLIC


def question(
    case_id: str | None, text: str = "Q?", dataset: str = "support-main", **fields: object
) -> BenchmarkQuestion:
    return BenchmarkQuestion(question=text, dataset=dataset, case_id=case_id, **fields)  # type: ignore[arg-type]


def stored(item: BenchmarkQuestion, question_id: int) -> BenchmarkQuestion:
    return replace(item, id=question_id, created_by=7)


class TestPlanQuestionSync:
    def test_matches_by_dataset_and_case_id_and_ignores_installation_fields(self) -> None:
        snapshot = [question("s-1", expected_answer="A")]
        current = [stored(question("s-1", expected_answer="A"), 10)]

        plan = plan_question_sync(snapshot, current)

        assert plan.in_sync
        assert plan.unchanged == 1

    def test_changed_fields_become_an_update_of_the_stored_id(self) -> None:
        snapshot = [question("s-1", text="New?", tags=["x"])]
        current = [stored(question("s-1", text="Old?"), 10)]

        plan = plan_question_sync(snapshot, current)

        assert [(item.question_id, item.changes) for item in plan.update] == [
            (10, {"question": "New?", "tags": ["x"]})
        ]

    def test_missing_question_is_created_and_database_only_one_is_kept(self) -> None:
        snapshot = [question("s-2")]
        current = [stored(question("s-1"), 10)]

        plan = plan_question_sync(snapshot, current)

        assert [q.case_id for q in plan.create] == ["s-2"]
        assert [q.id for q in plan.database_only] == [10]
        assert not plan.in_sync

    def test_same_case_id_in_another_dataset_is_another_question(self) -> None:
        plan = plan_question_sync([question("s-1", dataset="support-quick")], [stored(question("s-1"), 10)])

        assert len(plan.create) == 1
        assert len(plan.database_only) == 1

    @pytest.mark.parametrize("side", ["snapshot", "database"])
    def test_question_without_case_id_is_rejected_instead_of_matched_by_editable_text(
        self, side: str
    ) -> None:
        # Matching by text would turn a text edit into a new question on every other installation.
        snapshot = [question(None, text="Same?")] if side == "snapshot" else []
        current = [stored(question(None, text="Same?"), 3)] if side == "database" else []

        with pytest.raises(ValidationError, match="no case_id"):
            plan_question_sync(snapshot, current)

    def test_duplicate_keys_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="snapshot"):
            plan_question_sync([question("s-1"), question("s-1")], [])


class TestSnapshotFile:
    def test_round_trip_keeps_every_snapshot_field(self, tmp_path: Path) -> None:
        original = question(
            "s-1",
            expected_answer="A",
            source_hint="faq.txt",
            annotations={"required_facts": ["48 час"]},
            history=[{"role": "user", "content": "Привет"}],
            as_of_date=date(2025, 3, 1),
            tags=["t"],
            variant_of="s-0",
            question_style="short",
            notes="chat_log:2",
            is_active=False,
        )
        path = tmp_path / "questions.json"

        write_questions(path, [original])

        assert plan_question_sync(read_questions(path), [stored(original, 1)]).in_sync
        assert "Привет" in path.read_text(encoding="utf-8")

    def test_empty_values_are_omitted(self, tmp_path: Path) -> None:
        path = tmp_path / "questions.json"

        write_questions(path, [question("s-1")])

        record = json.loads(path.read_text(encoding="utf-8"))["questions"][0]
        assert record == {"dataset": "support-main", "case_id": "s-1", "question": "Q?", "is_active": True}

    def test_unknown_record_field_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "questions.json"
        path.write_text(
            json.dumps({"format": 1, "questions": [{"dataset": "d-x", "question": "Q", "id": 5}]})
        )

        with pytest.raises(ValidationError, match="unexpected"):
            read_questions(path)

    def test_missing_file_is_an_empty_snapshot(self, tmp_path: Path) -> None:
        assert read_questions(tmp_path / "questions.json") == []


class TestSeedManifest:
    def pack_with(self, tmp_path: Path, documents: dict[str, str], files: list[str]) -> DomainPack:
        (tmp_path / "corpus").mkdir()

        for name in files:
            (tmp_path / "corpus" / name).write_text("text", encoding="utf-8")

        (tmp_path / "seed.json").write_text(json.dumps({"space": "Space", "documents": documents}))
        return replace(NEUTRAL_PACK, key="demo", seed_dir=tmp_path)

    def test_manifest_lists_corpus_files_with_their_profiles(self, tmp_path: Path) -> None:
        seed = load_pack_seed(self.pack_with(tmp_path, {"a.txt": "general"}, ["a.txt"]))

        assert seed is not None
        assert [(d.file, d.domain) for d in seed.documents] == [("a.txt", "general")]
        assert seed.storage_key(seed.documents[0]) == "benchmark-corpus/demo/a.txt"
        assert seed.dataset_prefix == "demo-"

    @pytest.mark.parametrize(
        "documents, files",
        [({"a.txt": "general", "b.txt": "general"}, ["a.txt"]), ({"a.txt": "general"}, ["a.txt", "c.txt"])],
    )
    def test_corpus_and_manifest_must_agree(
        self, tmp_path: Path, documents: dict[str, str], files: list[str]
    ) -> None:
        with pytest.raises(ValidationError, match="corpus mismatch"):
            load_pack_seed(self.pack_with(tmp_path, documents, files))

    def test_pack_without_seed_has_none(self) -> None:
        assert load_pack_seed(NEUTRAL_PACK) is None


@pytest.mark.parametrize("pack_key", sorted(PACK_REGISTRATIONS))
def test_every_registered_pack_ships_a_consistent_seed(pack_key: str) -> None:
    pack = selected_pack(pack_key)
    seed = load_pack_seed(pack)

    assert seed is not None
    profiles = {profile.key for profile in pack.build_profiles(None)}
    assert {document.domain for document in seed.documents} <= profiles

    questions = read_questions(seed.questions_path)
    assert questions
    assert all(q.dataset.startswith(seed.dataset_prefix) for q in questions)
    # Raises on duplicate keys.
    plan_question_sync(questions, [])


class StoredQuestions(FakeBenchmarkQuestionRepository):
    def __init__(self, questions: list[BenchmarkQuestion]) -> None:
        super().__init__()
        self.items = questions

    async def list_items(self, **kwargs: object) -> list[BenchmarkQuestion]:
        del kwargs
        return list(self.items)

    async def bulk_create(self, questions: list[BenchmarkQuestion]) -> int:
        start = len(self.items) + 1
        self.items += [replace(q, id=start + offset) for offset, q in enumerate(questions)]
        return len(questions)

    async def update(self, question_id: int, **fields: object) -> BenchmarkQuestion | None:
        index = next(i for i, q in enumerate(self.items) if q.id == question_id)
        self.items[index] = replace(self.items[index], **fields)  # type: ignore[arg-type]
        return self.items[index]


class TestApplySync:
    @pytest.mark.asyncio
    async def test_apply_creates_updates_keeps_and_then_is_in_sync(self) -> None:
        repository = StoredQuestions(
            [
                stored(question("s-1", text="Old?"), 1),
                stored(question("local", text="Fresh edit?"), 2),
                stored(question("x-1", dataset="internal-main"), 3),
            ]
        )
        uow = FakeUnitOfWork()
        uow.benchmark_questions = repository
        service = BenchmarkQuestionService(FakeUnitOfWorkFactory(uow))
        snapshot = [question("s-1", text="New?"), question("s-2")]

        plan = await service.apply_sync(snapshot, "support-")

        assert (len(plan.create), len(plan.update), [q.case_id for q in plan.database_only]) == (
            1,
            1,
            ["local"],
        )
        texts = {q.case_id: q.question for q in repository.items}
        assert texts == {"s-1": "New?", "local": "Fresh edit?", "x-1": "Q?", "s-2": "Q?"}
        after = await service.plan_sync(snapshot, "support-")
        assert (after.create, after.update, after.unchanged) == ((), (), 2)

    @pytest.mark.asyncio
    async def test_snapshot_of_another_pack_is_rejected(self) -> None:
        service = BenchmarkQuestionService(FakeUnitOfWorkFactory())

        with pytest.raises(ValidationError, match="outside support-"):
            await service.apply_sync([question("x-1", dataset="internal-main")], "support-")


class TestCaseIdOnCreation:
    @pytest.mark.asyncio
    async def test_created_questions_get_a_case_id_unless_one_is_given(self) -> None:
        repository = StoredQuestions([])
        uow = FakeUnitOfWork()
        uow.benchmark_questions = repository
        service = BenchmarkQuestionService(FakeUnitOfWorkFactory(uow))

        await service.bulk_create(
            [
                BenchmarkQuestionCreateDTO(question="A?"),
                BenchmarkQuestionCreateDTO(question="B?", case_id="own"),
            ],
            created_by=1,
        )
        created = await service.create(BenchmarkQuestionCreateDTO(question="C?"), created_by=1)

        generated, given = repository.items
        assert generated.case_id is not None
        assert generated.case_id.startswith("q-")
        assert given.case_id == "own"
        assert created.case_id is not None
        assert created.case_id != generated.case_id


class TestCorpusReadiness:
    @pytest.mark.parametrize(
        "counts, ready",
        [
            ({"done": 3}, True),
            ({}, False),
            ({"done": 2, "indexing": 1}, False),
            ({"done": 2, "pending": 1}, False),
            ({"done": 2, "failed": 1}, False),
        ],
    )
    def test_corpus_is_searchable_only_when_every_document_is_indexed(
        self, counts: dict[str, int], ready: bool
    ) -> None:
        assert CorpusReadiness(1, counts).ready is ready

    @pytest.mark.asyncio
    async def test_readiness_counts_only_documents_of_the_space(self) -> None:
        factory = FakeUnitOfWorkFactory()
        documents = factory._uow.fake_documents
        await documents.save(
            Document(filename="a", visibility=PUBLIC, space_id=1, status=DocumentStatus.DONE)
        )
        await documents.save(
            Document(filename="b", visibility=PUBLIC, space_id=1, status=DocumentStatus.INDEXING)
        )
        await documents.save(
            Document(filename="c", visibility=PUBLIC, space_id=2, status=DocumentStatus.DONE)
        )

        assert (await corpus_readiness(factory, 1)).counts == {"done": 1, "indexing": 1}
        assert (await corpus_readiness(factory, 2)).ready


class FakeStorage:
    def __init__(self, files: dict[str, bytes], tmp_path: Path) -> None:
        self.files = files
        self.tmp_path = tmp_path

    async def get_file_info(self, key: str) -> object | None:
        return object() if key in self.files else None

    async def download_to_temp(self, key: str) -> Path:
        path = self.tmp_path / f"download-{len(list(self.tmp_path.iterdir()))}"
        path.write_bytes(self.files[key])
        return path


class TestManualCopy:
    def infra(self, tmp_path: Path, files: dict[str, bytes]) -> Any:
        return SimpleNamespace(ml=SimpleNamespace(storage=FakeStorage(files, tmp_path)))

    def document(self, status: DocumentStatus) -> Document:
        return Document(
            id=3, filename="a.docx", visibility=PUBLIC, status=status, source_path="uploads/a.docx"
        )

    @pytest.mark.asyncio
    async def test_indexed_identical_copy_stands_in_for_the_seed_file(self, tmp_path: Path) -> None:
        infra = self.infra(tmp_path, {"uploads/a.docx": b"same"})

        assert await manual_copy_conflict(infra, self.document(DocumentStatus.DONE), b"same") is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status, stored, reason",
        [
            (DocumentStatus.FAILED, b"same", "status is failed"),
            (DocumentStatus.INDEXING, b"same", "status is indexing"),
            (DocumentStatus.DONE, b"older edition", "content differs"),
        ],
    )
    async def test_failed_unfinished_or_different_copy_is_a_conflict(
        self, tmp_path: Path, status: DocumentStatus, stored: bytes, reason: str
    ) -> None:
        infra = self.infra(tmp_path, {"uploads/a.docx": stored})

        conflict = await manual_copy_conflict(infra, self.document(status), b"same")

        assert conflict is not None
        assert reason in conflict
