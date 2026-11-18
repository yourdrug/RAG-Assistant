"""Characterization tests for DocumentProcessor -- baseline BEFORE refactoring.

Pins the observable behavior of the service (parse → quality → classify →
split → versioning → persist → cleanup) so the extraction of
document_quality / domain_classification / document_persistence modules and the
ProcessingContext refactor cannot silently change:

- exact warning_message strings and their join order (quality → ambiguous → versioning);
- transaction ordering (status write → chunks and replacement in one UoW);
- outbox operations (exactly one DELETE_BY_DOCUMENT per replaced document);
- deferred S3 cleanup (only committed replacements may delete old files);
- metrics calls and temp-file removal on every exit path.

If a refactor intentionally changes one of these, update the assertion
deliberately -- never delete the test.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import threading
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Unpack

import pytest

from application.documents import DocumentProcessor
from application.documents.dto import DocumentProcessingDependencies, ProcessDocumentCommand
from domain.ingestion.preview import PDFQualityReport
from domain.ingestion.profiles.registry import ClassificationResult
from domain.ingestion.raw_document import RawDocument
from domain.knowledge.document import Document
from domain.knowledge.enums import DocumentStatus, DomainKey
from domain.knowledge.index_sync import OutboxOperation
from infrastructure.ingestion.document_parser_adapter import IngestionDocumentSplitter
from infrastructure.ml.langchain_document_parser import LangchainDocumentSplitter
from tests.fakes import FakeDocumentRepository, FakeUnitOfWorkFactory, require_present


if TYPE_CHECKING:
    from pathlib import Path

    from application.ingestion import FileMeta, SplitContext
    from domain.ingestion.profiles.protocol import DocumentProfile
    from domain.knowledge.repositories.document import DocumentStatusUpdate
    from tests.fakes import FakeUnitOfWork

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class StubFileStorage:
    def __init__(self, tmp_dir: Path) -> None:
        self._tmp_dir = tmp_dir
        self.downloads: list[str] = []
        self.deleted: list[str] = []
        self._n = 0

    async def download_to_temp(self, key: str) -> Any:
        self._n += 1
        path = self._tmp_dir / f"temp_{self._n}.bin"
        path.write_bytes(b"stub-file-content")
        self.downloads.append(key)
        return path

    async def delete_file(self, key: str) -> None:
        self.deleted.append(key)


class StubParser:
    def __init__(self, docs: Any, error: Exception | None = None) -> None:
        self._docs = docs
        self._error = error
        self.calls: list[Any] = []

    def parse(self, file_path: Path, meta: FileMeta) -> list[Any]:
        _ = meta
        self.calls.append(file_path)

        if self._error is not None:
            raise self._error

        return [RawDocument(page_content=d.page_content, metadata=dict(d.metadata)) for d in self._docs]


class StubSplitter:
    """Copies doc metadata into chunks so metadata enrichment stays observable."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def split(self, documents: Any, context: SplitContext) -> list[Any]:
        self.calls.append((tuple(documents), context.domain))
        return [RawDocument(page_content=d.page_content, metadata=dict(d.metadata)) for d in documents]


class StubExtractor:
    def extract_date_from_filename(self, filename: str) -> str | None:
        _ = filename
        return "2026-01-15"


class StubPDFQualityAssessor:
    def __init__(self, report: PDFQualityReport) -> None:
        self._report = report
        self.calls: list[Any] = []

    def assess(self, pdf_path: Any, documents: Any) -> PDFQualityReport:
        self.calls.append((pdf_path, documents))
        return self._report


class StubTextQualityAssessor:
    def __init__(self, marker: str = "мусор") -> None:
        self._marker = marker

    def is_garbled(self, text: str) -> bool:
        return self._marker in text


class StubSettings:
    def get(self, key: str, domain_key: str = "") -> str:
        _ = domain_key, key
        return "0"


class RecordingMetrics:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def inc_chunks(self, count: int) -> None:
        self.calls.append(("inc_chunks", count))

    def inc_documents(self, status: str) -> None:
        self.calls.append(("inc_documents", status))

    def observe_duration(self, status: str, seconds: float) -> None:
        _ = seconds
        self.calls.append(("observe_duration", status))

    def observe_pdf_pages(self, quality: str, count: int) -> None:
        self.calls.append(("observe_pdf_pages", quality, count))

    def observe_pdf_bad_ratio(self, ratio: float) -> None:
        self.calls.append(("observe_pdf_bad_ratio", ratio))

    def observe_domain_classification(self, domain: str, level: str, confidence: float) -> None:
        self.calls.append(("observe_domain_classification", domain, level, confidence))

    def inc_domain_ambiguous(self, candidates: str) -> None:
        self.calls.append(("inc_domain_ambiguous", candidates))

    def count(self, name: str) -> int:
        return sum(1 for c in self.calls if c[0] == name)

    def has(self, *expected: Any) -> bool:
        return any(c[: len(expected)] == expected for c in self.calls)


class RecordingDocumentRepository(FakeDocumentRepository):
    """FakeDocumentRepository + status/domain call recording."""

    def __init__(self) -> None:
        super().__init__()
        self.status_calls: list[dict[str, Any]] = []
        self.domain_calls: list[tuple[int, str]] = []
        self.get_calls = 0
        self.fail_failed_status = False

    async def get_by_id(self, document_id: int, *, for_update: bool = False) -> Document | None:
        _ = for_update
        doc_id = document_id
        self.get_calls += 1
        return await super().get_by_id(doc_id)

    async def set_domain(self, document_id: int, doc_domain: str) -> None:
        self.domain_calls.append((document_id, doc_domain))
        doc = self._documents.get(document_id)

        if doc is not None:
            doc.doc_domain = DomainKey(doc_domain)

    async def update_status(self, document_id: int, **kwargs: Unpack[DocumentStatusUpdate]) -> bool:
        status = kwargs.get("status")

        if self.fail_failed_status and status == DocumentStatus.FAILED.value:
            msg = "status write failed"
            raise RuntimeError(msg)

        self.status_calls.append({"document_id": document_id, "status": status, **kwargs})
        doc = self._documents.get(document_id)

        if doc is not None:
            doc.status = DocumentStatus(require_present(status))

            if "error" in kwargs:
                doc.error_message = kwargs["error"]

            if "warning" in kwargs:
                doc.warning_message = kwargs["warning"]

            if "quality_score" in kwargs:
                doc.quality_score = kwargs["quality_score"]

            if "chunks" in kwargs:
                doc.chunks = kwargs["chunks"]

        return doc is not None


class VanishingDocumentRepository(RecordingDocumentRepository):
    """get_by_id for one doc id returns None after N successful calls.

    Simulates the document row being deleted mid-flight by another session.
    """

    def __init__(self, vanish_id: int, vanish_after: int) -> None:
        super().__init__()
        self._vanish_id = vanish_id
        self._vanish_after = vanish_after
        self._seen = 0

    async def get_by_id(self, document_id: int, *, for_update: bool = False) -> Document | None:
        _ = for_update
        doc_id = document_id

        if doc_id == self._vanish_id:
            if self._seen >= self._vanish_after:
                self.get_calls += 1
                return None

            self._seen += 1

        return await super().get_by_id(doc_id)


class FakeDomainRegistry:
    def __init__(
        self,
        profiles: dict[str, DocumentProfile] | None = None,
        classify_result: ClassificationResult | None = None,
    ) -> None:
        self._profiles = profiles or {}
        self._classify_result = classify_result
        self.classify_calls: list[str] = []

    def get(self, key: str) -> DocumentProfile:
        return self._profiles[key]  # KeyError, like the real registry

    def classify(
        self, text: str, *, settings: object, prior_domain: str | None = None
    ) -> ClassificationResult:
        _ = settings, prior_domain
        self.classify_calls.append(text)

        if self._classify_result is None:
            msg = "classify_result not configured"
            raise RuntimeError(msg)

        return self._classify_result


def make_profile(key: str = "general", versioned: bool = False) -> SimpleNamespace:
    return SimpleNamespace(key=key, is_versioned=versioned)


def _default_docs() -> list[RawDocument]:
    return [
        RawDocument(page_content="Договор аренды офисного помещения. Стороны договорились о нижеследующем."),
        RawDocument(page_content="Раздел 1. Общие положения аренды помещения и сроки."),
    ]


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def build_harness(
    tmp_path: Path, *, docs: Any = None, parse_error: Any = None, repo: Any = None, **overrides: Any
) -> SimpleNamespace:
    uow_factory = overrides.pop("uow_factory", None) or FakeUnitOfWorkFactory()
    pack_hooks = overrides.pop("pack_hooks", None)
    repo = repo or RecordingDocumentRepository()
    uow_factory._uow.documents = repo

    harness = SimpleNamespace(
        uow_factory=uow_factory,
        uow=uow_factory._uow,
        repo=repo,
        storage=StubFileStorage(tmp_path),
        metrics=RecordingMetrics(),
        parser=StubParser(docs if docs is not None else _default_docs(), error=parse_error),
        splitter=overrides.pop("splitter", StubSplitter()),
        pdf_assessor=StubPDFQualityAssessor(
            overrides.pop(
                "pdf_report",
                PDFQualityReport(total_pages=10, n_ok=10, n_missing=0, n_garbled=0, bad_ratio=0.0),
            )
        ),
        text_quality=StubTextQualityAssessor(),
        registry=overrides.pop("registry", None),
        temp_paths=[],
    )
    harness.processor = DocumentProcessor(
        uow_factory=uow_factory,
        file_storage=harness.storage,
        dependencies=DocumentProcessingDependencies(
            document_parser=harness.parser,
            document_splitter=harness.splitter,
            content_extractor=StubExtractor(),
            pdf_quality_assessor=harness.pdf_assessor,
            text_quality_assessor=harness.text_quality,
            metrics=harness.metrics,
        ),
        domain_registry=harness.registry,
        domain_settings=StubSettings() if harness.registry is not None else None,
        pack_hooks=pack_hooks,
    )
    return harness


async def seed_document(repo: RecordingDocumentRepository, **kwargs: Any) -> Document:
    kwargs.setdefault("filename", "report.pdf")
    kwargs.setdefault("source_path", "uploads/report.pdf")
    return await repo.save(Document(**kwargs))


def outbox_ops(uow: FakeUnitOfWork) -> list[tuple[OutboxOperation, int]]:
    return [(e.operation, e.aggregate_id) for e in uow.fake_vector_outbox._entries.values()]


def upsert_points(uow: FakeUnitOfWork) -> list[dict[str, Any]]:
    points = []

    for e in uow.fake_vector_outbox._entries.values():
        if e.operation == OutboxOperation.UPSERT_CHUNKS:
            points.extend(e.payload["points"])

    return points


async def run(harness: Any, doc: Any, **kwargs: Any) -> None:
    kwargs.setdefault("storage_key", "uploads/report.pdf")
    kwargs.setdefault("original_filename", doc.filename)
    kwargs.setdefault("visibility", "internal_public")
    kwargs.setdefault("owner_id", 1)
    kwargs.setdefault("group_id", None)
    kwargs.setdefault("replace_id", None)
    kwargs.setdefault("doc_domain", "general")
    await harness.processor.process(ProcessDocumentCommand(document_id=doc.id, **kwargs))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_happy_path_indexes_document(tmp_path: Path) -> None:
    h = build_harness(tmp_path)
    doc = await seed_document(h.repo)

    await run(h, doc)

    assert [c["status"] for c in h.repo.status_calls] == ["processing", "indexing"]
    assert h.repo.status_calls[1]["chunks"] == 2
    assert h.repo.domain_calls == [(doc.id, "general")]
    assert outbox_ops(h.uow) == [(OutboxOperation.UPSERT_CHUNKS, doc.id)]
    assert len(upsert_points(h.uow)) == 2
    assert h.metrics.count("inc_documents") == 1
    assert h.metrics.has("inc_documents", "indexing")
    assert h.metrics.has("inc_chunks", 2)
    assert h.metrics.count("observe_duration") == 1
    assert h.storage.deleted == []
    # temp file is removed in finalize_processing
    assert list(tmp_path.iterdir()) == []  # noqa: ASYNC240 - Small local fixture I/O in the test setup.


@pytest.mark.asyncio
async def test_empty_parse_marks_failed(tmp_path: Path) -> None:
    h = build_harness(tmp_path, docs=[])
    doc = await seed_document(h.repo)

    with pytest.raises(RuntimeError, match="Текст не извлечён"):
        await run(h, doc)

    expected_error = "Текст не извлечён — документ похож на скан, и OCR не смог распознать содержимое."
    assert len(h.repo.status_calls) == 2
    assert h.repo.status_calls[1]["status"] == "failed"
    assert h.repo.status_calls[1]["error"] == expected_error
    assert h.metrics.has("inc_documents", "failed")
    assert outbox_ops(h.uow) == []
    assert h.repo.domain_calls == []
    assert list(tmp_path.iterdir()) == []  # noqa: ASYNC240 - Small local fixture I/O in the test setup.


@pytest.mark.asyncio
async def test_parse_failure_marks_failed_and_cleans_temp(tmp_path: Path) -> None:
    h = build_harness(tmp_path, parse_error=RuntimeError("boom"))
    doc = await seed_document(h.repo)

    with pytest.raises(RuntimeError, match="boom"):
        await run(h, doc)

    assert h.repo.status_calls[-1]["status"] == "failed"
    assert h.repo.status_calls[-1]["error"] == "boom"
    assert h.metrics.has("inc_documents", "failed")
    assert list(tmp_path.iterdir()) == []  # noqa: ASYNC240 - Small local fixture I/O in the test setup.


@pytest.mark.asyncio
async def test_status_write_failure_does_not_mask_processing_error(tmp_path: Path) -> None:
    """The secondary status-write error is logged while the processing error propagates."""
    h = build_harness(tmp_path, parse_error=RuntimeError("boom"))
    h.repo.fail_failed_status = True
    doc = await seed_document(h.repo)

    with pytest.raises(RuntimeError, match="boom"):
        await run(h, doc)

    assert h.metrics.has("inc_documents", "failed")
    assert list(tmp_path.iterdir()) == []  # noqa: ASYNC240 - Small local fixture I/O in the test setup.


@pytest.mark.asyncio
async def test_low_quality_pdf_warning_and_metrics(tmp_path: Path) -> None:
    report = PDFQualityReport(total_pages=10, n_ok=5, n_missing=3, n_garbled=2, bad_ratio=0.5)
    h = build_harness(tmp_path, pdf_report=report)
    doc = await seed_document(h.repo, filename="scan.pdf", source_path="uploads/scan.pdf")

    await run(h, doc, storage_key="uploads/scan.pdf", original_filename="scan.pdf")

    expected_warning = (
        "Низкое качество распознавания: 3 стр. без текста, 2 стр. с мусорным текстом из 10. "
        "Рекомендуется проверить документ (task pdf:diag) и переиндексировать "
        "после конвертации или ручной вычитки."
    )
    final = h.repo.status_calls[-1]
    assert final["status"] == "indexing"
    assert final["warning"] == expected_warning
    assert final["quality_score"] == 0.5
    assert h.metrics.has("observe_pdf_pages", "ok", 5)
    assert h.metrics.has("observe_pdf_pages", "missing", 3)
    assert h.metrics.has("observe_pdf_pages", "garbled", 2)
    assert h.metrics.has("observe_pdf_bad_ratio", 0.5)


@pytest.mark.asyncio
async def test_docx_garbled_warning_without_quality_report(tmp_path: Path) -> None:
    """Non-PDF branch: no report object -> quality_score is explicitly None."""
    docs = [
        RawDocument(page_content="текст нормального качества на этой странице документа"),
        RawDocument(page_content="текст нормального качества на этой странице документа"),
        RawDocument(page_content="текст нормального качества на этой странице документа"),
        RawDocument(page_content="мусор мусор мусор мусор мусор мусор"),
        RawDocument(page_content="мусор мусор мусор мусор мусор мусор"),
    ]
    h = build_harness(tmp_path, docs=docs)
    doc = await seed_document(h.repo, filename="memo.docx", source_path="uploads/memo.docx")

    await run(h, doc, storage_key="uploads/memo.docx", original_filename="memo.docx")

    expected_warning = (
        "Низкое качество извлечения текста: 2 стр. с мусорным текстом, "
        "0 пустых стр. из 5. Рекомендуется проверить документ."
    )
    final = h.repo.status_calls[-1]
    assert final["warning"] == expected_warning
    assert final["quality_score"] is None
    assert h.pdf_assessor.calls == []
    assert not h.metrics.has("observe_pdf_bad_ratio", 0.0)


@pytest.mark.asyncio
async def test_very_low_char_count_warning(tmp_path: Path) -> None:
    docs = [RawDocument(page_content="abc")]
    h = build_harness(tmp_path, docs=docs)
    doc = await seed_document(h.repo, filename="note.txt", source_path="uploads/note.txt")

    await run(h, doc, storage_key="uploads/note.txt", original_filename="note.txt")

    expected = "Документ содержит очень мало текста (3 символов). Возможно, это скан или повреждённый файл."
    assert h.repo.status_calls[-1]["warning"] == expected


@pytest.mark.asyncio
async def test_ambiguous_classification_appends_warning(tmp_path: Path) -> None:
    """quality warning and ambiguous warning are joined with a single newline."""
    docs = [
        RawDocument(page_content="текст нормального качества на этой странице документа"),
        RawDocument(page_content="текст нормального качества на этой странице документа"),
        RawDocument(page_content="текст нормального качества на этой странице документа"),
        RawDocument(page_content="мусор мусор мусор мусор мусор мусор"),
        RawDocument(page_content="мусор мусор мусор мусор мусор мусор"),
    ]
    registry = FakeDomainRegistry(
        profiles={"legal": make_profile("legal", versioned=True)},
        classify_result=ClassificationResult(
            domain_key="legal",
            confidence=0.4,
            is_ambiguous=True,
            is_low_signal=False,
            candidate_scores={"legal": 6.0, "general": 4.0, "__low_signal": 9.0},
        ),
    )
    h = build_harness(tmp_path, docs=docs, registry=registry)
    doc = await seed_document(h.repo, filename="memo.docx", source_path="uploads/memo.docx")

    await run(h, doc, storage_key="uploads/memo.docx", original_filename="memo.docx", doc_domain=None)

    quality_warning = (
        "Низкое качество извлечения текста: 2 стр. с мусорным текстом, "
        "0 пустых стр. из 5. Рекомендуется проверить документ."
    )
    ambiguous_warning = (
        "Классификация домена неоднозначна (legal=6.0, general=4.0) — требуется ручная проверка"
    )
    final = h.repo.status_calls[-1]
    assert final["warning"] == f"{quality_warning}\n{ambiguous_warning}"
    assert h.repo.domain_calls == [(doc.id, "legal")]
    assert h.metrics.has("inc_domain_ambiguous", "general,legal")
    assert h.metrics.has("observe_domain_classification", "legal", "document")


@pytest.mark.asyncio
async def test_general_fallback_when_registry_missing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    h = build_harness(tmp_path, docs=[RawDocument(page_content="Федеральный закон")])
    doc = await seed_document(h.repo)

    await run(h, doc, doc_domain=None)

    assert h.repo.domain_calls == [(doc.id, "general")]
    assert "Domain registry or settings unavailable -- using general domain" in caplog.text
    assert outbox_ops(h.uow) == [(OutboxOperation.UPSERT_CHUNKS, doc.id)]


@pytest.mark.asyncio
@pytest.mark.parametrize("ingestion_key", [None, "docs/shared.pdf"])
async def test_replace_non_versioned_deletes_old_document(tmp_path: Path, ingestion_key: str | None) -> None:
    registry = FakeDomainRegistry(
        profiles={"general": make_profile("general", versioned=False)},
        classify_result=ClassificationResult(
            domain_key="general", confidence=0.9, is_ambiguous=False, is_low_signal=False, candidate_scores={}
        ),
    )
    h = build_harness(tmp_path, registry=registry)
    old = await seed_document(
        h.repo, filename="old.pdf", source_path="uploads/old.pdf", ingestion_key=ingestion_key
    )
    new = await seed_document(h.repo, filename="new.pdf", source_path="uploads/new.pdf")

    await run(
        h,
        new,
        storage_key="uploads/new.pdf",
        original_filename="new.pdf",
        replace_id=old.id,
        doc_domain=None,
    )

    # Replacement commits together with the new chunks.
    assert await h.repo.get_by_id(old.id) is None
    # deferred S3 cleanup runs once, after the transaction
    assert h.storage.deleted == ([] if ingestion_key else ["uploads/old.pdf"])
    assert outbox_ops(h.uow) == [
        (OutboxOperation.UPSERT_CHUNKS, new.id),
        (OutboxOperation.DELETE_BY_DOCUMENT, old.id),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "new_filename, expected_filename",
    [("old(1).pdf", "old.pdf"), ("other.pdf", "other.pdf")],
)
async def test_replacement_restores_name_held_by_replaced_document(
    tmp_path: Path, new_filename: str, expected_filename: str
) -> None:
    registry = FakeDomainRegistry(
        profiles={"general": make_profile("general", versioned=False)},
        classify_result=ClassificationResult(
            domain_key="general", confidence=0.9, is_ambiguous=False, is_low_signal=False, candidate_scores={}
        ),
    )
    h = build_harness(tmp_path, registry=registry)
    old = await seed_document(h.repo, filename="old.pdf", source_path="uploads/old.pdf")
    new = await seed_document(h.repo, filename=new_filename, source_path="uploads/new.pdf")

    await run(h, new, storage_key="uploads/new.pdf", replace_id=old.id, doc_domain=None)

    restored = await h.repo.get_by_id(new.id)
    assert restored is not None
    assert restored.filename == expected_filename
    assert restored.source_path == "uploads/new.pdf"
    assert {point["metadata"]["source"] for point in upsert_points(h.uow)} == {expected_filename}


@pytest.mark.asyncio
async def test_document_deleted_before_persist_aborts(tmp_path: Path) -> None:
    """current_doc is None -> early abort: no chunks, no outbox, status failed."""
    h = build_harness(tmp_path, repo=VanishingDocumentRepository(vanish_id=1, vanish_after=1))
    doc = await seed_document(h.repo)

    await run(h, doc)

    # abort is silent: no FAILED write, the row stays in "processing"
    # (reconciled later by mark_stuck_processing_failed)
    assert h.repo.get_calls == 2
    assert [c["status"] for c in h.repo.status_calls] == ["processing"]
    assert h.repo.domain_calls == []
    assert outbox_ops(h.uow) == []
    assert h.metrics.has("inc_documents", "failed")


@pytest.mark.asyncio
async def test_document_deleted_before_processing_skips_storage(tmp_path: Path) -> None:
    h = build_harness(tmp_path, repo=VanishingDocumentRepository(vanish_id=1, vanish_after=0))
    doc = await seed_document(h.repo)
    await run(h, doc)
    assert h.repo.status_calls == []
    assert h.storage.downloads == []
    assert h.storage.deleted == []
    assert outbox_ops(h.uow) == []


@pytest.mark.asyncio
async def test_document_deleted_inside_persist_preserves_replacement_target(tmp_path: Path) -> None:
    """An aborted replacement must preserve both the old row and its source file."""
    registry = FakeDomainRegistry(
        profiles={"general": make_profile("general", versioned=False)},
        classify_result=ClassificationResult(
            domain_key="general", confidence=0.9, is_ambiguous=False, is_low_signal=False, candidate_scores={}
        ),
    )
    repo = VanishingDocumentRepository(vanish_id=2, vanish_after=1)
    h = build_harness(tmp_path, registry=registry, repo=repo)
    old = await seed_document(h.repo, filename="old.pdf", source_path="uploads/old.pdf")
    new = await seed_document(h.repo, filename="new.pdf", source_path="uploads/new.pdf")

    await run(
        h,
        new,
        storage_key="uploads/new.pdf",
        original_filename="new.pdf",
        replace_id=old.id,
        doc_domain=None,
    )

    assert repo.get_calls == 2
    assert [c["status"] for c in h.repo.status_calls] == ["processing"]
    assert h.repo.domain_calls == []
    assert outbox_ops(h.uow) == []
    assert h.storage.deleted == []
    assert await h.repo.get_by_id(old.id) is not None


@pytest.mark.asyncio
async def test_section_prefix_added_to_chunks(tmp_path: Path) -> None:

    docs = [RawDocument(page_content="Текст раздела", metadata={"section": "Глава 1"})]
    h = build_harness(tmp_path, docs=docs, splitter=IngestionDocumentSplitter(LangchainDocumentSplitter()))
    doc = await seed_document(h.repo)

    await run(h, doc)

    points = upsert_points(h.uow)
    assert points[0]["page_content"] == "Текст раздела"
    assert points[0]["embed_text"] == "Глава 1\n\nТекст раздела"
    assert "body_text" not in points[0]["metadata"]
    # _attach_metadata_to_docs runs before split -> source lands on every chunk
    assert points[0]["metadata"]["source"] == "report.pdf"
    assert points[0]["metadata"]["doc_date"] == "2026-01-15"


@pytest.mark.asyncio
async def test_concurrent_processing_bounds_parser_and_keeps_run_state_separate(tmp_path: Path) -> None:  # noqa: PLR0915 - One scenario verifies the complete transaction/lifecycle.
    h = build_harness(tmp_path)
    first = await seed_document(h.repo, filename="first.pdf")
    second = await seed_document(h.repo, filename="second.pdf")
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    downloaded_second = asyncio.Event()
    release = threading.Event()
    original_parse = h.parser.parse
    thread_ids = []

    def blocking_parse(path: Any, meta: Any) -> Any:
        thread_ids.append(threading.get_ident())

        if len(thread_ids) == 1:
            loop.call_soon_threadsafe(entered.set)

            if not release.wait(timeout=5):
                msg = "Parser was not released"
                raise RuntimeError(msg)

        return original_parse(path, meta)

    original_download = h.storage.download_to_temp

    async def download(key: str) -> Any:
        path = await original_download(key)

        if key == "second":
            downloaded_second.set()

        return path

    h.parser.parse = blocking_parse
    h.storage.download_to_temp = download
    tasks = [asyncio.create_task(run(h, first, storage_key="first"))]
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        tasks.append(asyncio.create_task(run(h, second, storage_key="second")))
        await asyncio.wait_for(downloaded_second.wait(), timeout=5)
        # Second download reached the parsing step while the first parser is blocked.
        assert len(thread_ids) == 1
    finally:
        release.set()
        await asyncio.gather(*tasks)
    assert len(thread_ids) == 2
    assert all(tid != threading.get_ident() for tid in thread_ids)
    assert h.metrics.count("inc_documents") == 2
    sources = {p["metadata"]["source"] for p in upsert_points(h.uow)}
    assert sources == {"first.pdf", "second.pdf"}
    assert list(tmp_path.iterdir()) == []  # noqa: ASYNC240 - Small local fixture I/O in the test setup.


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", ["split", "commit"])
async def test_failed_replacement_preserves_old_document_and_file(tmp_path: Path, failure_stage: str) -> None:
    h = build_harness(tmp_path)
    old = await seed_document(h.repo, filename="old.pdf", source_path="uploads/old.pdf")
    new = await seed_document(h.repo, filename="new.pdf", source_path="uploads/new.pdf")

    if failure_stage == "split":

        def fail_split(*args: Any) -> None:
            msg = "replacement failed"
            raise RuntimeError(msg)

        h.splitter.split = fail_split
    else:
        original_create = h.uow_factory.create

        @asynccontextmanager
        async def fail_commit(master: bool = False) -> Any:
            async with original_create(master=master) as uow:
                yield uow

                if any(
                    e.operation == OutboxOperation.DELETE_BY_DOCUMENT
                    for e in uow.fake_vector_outbox._entries.values()
                ):
                    msg = "replacement failed"
                    raise RuntimeError(msg)

        h.uow_factory.create = fail_commit

    with pytest.raises(RuntimeError, match="replacement failed"):
        await run(h, new, replace_id=old.id)

    assert await h.repo.get_by_id(old.id) is not None
    assert (await h.repo.get_by_id(new.id)).status == DocumentStatus.FAILED
    assert h.storage.deleted == []
    assert outbox_ops(h.uow) == []
