"""Characterization tests for DocumentProcessor -- baseline BEFORE refactoring.

Pins the observable behavior of the god-service (parse → quality → classify →
conflict → split → versioning → persist → cleanup) so the extraction of
document_quality / domain_classification / document_persistence modules and the
ProcessingContext refactor cannot silently change:

- exact warning_message strings and their join order (quality → ambiguous → versioning);
- transaction ordering (status write → conflict resolve → persist in one UoW);
- outbox operations (exactly one DELETE_BY_DOCUMENT per replaced document);
- deferred S3 cleanup (delete_file runs after the persist transaction on EVERY
  exit path, including mid-flight aborts);
- metrics calls and temp-file removal on every exit path.

If a refactor intentionally changes one of these, update the assertion
deliberately -- never delete the test.
"""

from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager
from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from application.dto.versioning_dto import VersioningPlan, VersioningResult
from application.services.document_processor import DocumentProcessor
from domain.domain_profile.registry import ClassificationResult
from domain.entities.document import Document
from domain.entities.raw_document import RawDocument
from domain.entities.vector_outbox_entry import OutboxOperation
from domain.value_objects.document_status import DocumentStatus
from domain.value_objects.pdf_quality_report import PDFQualityReport
from fakes import FakeDocumentRepository, FakeUnitOfWorkFactory

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class StubFileStorage:
    def __init__(self, tmp_dir) -> None:
        self._tmp_dir = tmp_dir
        self.downloads: list[str] = []
        self.deleted: list[str] = []
        self._n = 0

    async def download_to_temp(self, key: str):
        self._n += 1
        path = self._tmp_dir / f"temp_{self._n}.bin"
        path.write_bytes(b"stub-file-content")
        self.downloads.append(key)
        return path

    async def delete_file(self, key: str) -> None:
        self.deleted.append(key)


class StubParser:
    def __init__(self, docs, error: Exception | None = None) -> None:
        self._docs = docs
        self._error = error
        self.calls: list = []

    def parse(self, file_path) -> list:
        self.calls.append(file_path)
        if self._error is not None:
            raise self._error
        return [RawDocument(page_content=d.page_content, metadata=dict(d.metadata)) for d in self._docs]


class StubSplitter:
    """Copies doc metadata into chunks so metadata enrichment stays observable."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def split(self, documents, domain: str = "general") -> list:
        self.calls.append((tuple(documents), domain))
        return [RawDocument(page_content=d.page_content, metadata=dict(d.metadata)) for d in documents]


class StubExtractor:
    def extract_date_from_filename(self, filename: str) -> str | None:
        return "2026-01-15"


class StubPDFQualityAssessor:
    def __init__(self, report: PDFQualityReport) -> None:
        self._report = report
        self.calls: list = []

    def assess(self, pdf_path, documents) -> PDFQualityReport:
        self.calls.append((pdf_path, documents))
        return self._report


class StubTextQualityAssessor:
    def __init__(self, marker: str = "мусор") -> None:
        self._marker = marker

    def is_garbled(self, text: str) -> bool:
        return self._marker in text


class StubSettings:
    def get(self, key: str, domain_key: str = "") -> str:
        return "0"


class RecordingMetrics:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def inc_chunks(self, count: int) -> None:
        self.calls.append(("inc_chunks", count))

    def inc_documents(self, status: str) -> None:
        self.calls.append(("inc_documents", status))

    def observe_duration(self, status: str, seconds: float) -> None:
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

    def has(self, *expected) -> bool:
        return any(c[: len(expected)] == expected for c in self.calls)


class RecordingDocumentRepository(FakeDocumentRepository):
    """FakeDocumentRepository + status/domain call recording."""

    def __init__(self) -> None:
        super().__init__()
        self.status_calls: list[dict] = []
        self.domain_calls: list[tuple[int, str]] = []
        self.get_calls = 0
        self.fail_failed_status = False

    async def get_by_id(self, doc_id: int):
        self.get_calls += 1
        return await super().get_by_id(doc_id)

    async def set_domain(self, document_id: int, doc_domain: str) -> None:
        self.domain_calls.append((document_id, doc_domain))
        doc = self._documents.get(document_id)
        if doc is not None:
            doc.doc_domain = str(doc_domain)

    async def update_status(self, document_id: int, status: str, **kwargs) -> None:
        if self.fail_failed_status and status == DocumentStatus.FAILED.value:
            raise RuntimeError("status write failed")
        self.status_calls.append({"document_id": document_id, "status": status, **kwargs})
        doc = self._documents.get(document_id)
        if doc is not None:
            doc.status = DocumentStatus(status)
            if "error" in kwargs:
                doc.error_message = kwargs["error"]
            if "warning" in kwargs:
                doc.warning_message = kwargs["warning"]
            if "quality_score" in kwargs:
                doc.quality_score = kwargs["quality_score"]
            if "chunks" in kwargs:
                doc.chunks = kwargs["chunks"]


class VanishingDocumentRepository(RecordingDocumentRepository):
    """get_by_id for one doc id returns None after N successful calls.

    Simulates the document row being deleted mid-flight by another session.
    """

    def __init__(self, vanish_id: int, vanish_after: int) -> None:
        super().__init__()
        self._vanish_id = vanish_id
        self._vanish_after = vanish_after
        self._seen = 0

    async def get_by_id(self, doc_id: int):
        if doc_id == self._vanish_id:
            if self._seen >= self._vanish_after:
                self.get_calls += 1
                return None
            self._seen += 1
        return await super().get_by_id(doc_id)


class FakeActVersioningService:
    def __init__(self, result: VersioningResult | None = None) -> None:
        self.calls: list[dict] = []
        self.invalidated_acts: list[int] = []
        self.invalidated_documents: list[int] = []
        self._result = result or VersioningResult(
            domain_metadata=None, act_version_id=None, act_id=None, effective_from=None, warning=None
        )

    async def process_document_versioning(
        self, profile, document_id: int, full_text: str
    ) -> VersioningResult:
        self.calls.append({"profile": profile, "document_id": document_id, "full_text": full_text})
        return self._result

    async def invalidate_act_answers(self, act_id: int) -> None:
        self.invalidated_acts.append(act_id)

    async def invalidate_document_answers(self, document_id: int) -> None:
        self.invalidated_documents.append(document_id)


class FakeDomainRegistry:
    def __init__(
        self, profiles: dict | None = None, classify_result: ClassificationResult | None = None
    ) -> None:
        self._profiles = profiles or {}
        self._classify_result = classify_result
        self.classify_calls: list[str] = []

    def get(self, key: str):
        return self._profiles[key]  # KeyError, like the real registry

    def classify(self, text: str, *, settings, prior_domain: str | None = None) -> ClassificationResult:
        self.classify_calls.append(text)
        if self._classify_result is None:
            raise RuntimeError("classify_result not configured")
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


def build_harness(tmp_path, *, docs=None, parse_error=None, repo=None, **overrides):
    uow_factory = FakeUnitOfWorkFactory()
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
        act_versioning=overrides.pop("act_versioning", None),
        temp_paths=[],
    )
    harness.processor = DocumentProcessor(
        uow_factory=uow_factory,
        vector_store_repo=MagicMock(),
        file_storage=harness.storage,
        document_parser=harness.parser,
        document_splitter=harness.splitter,
        content_extractor=StubExtractor(),
        pdf_quality_assessor=harness.pdf_assessor,
        text_quality_assessor=harness.text_quality,
        metrics=harness.metrics,
        domain_marker_threshold=1.0,
        domain_registry=harness.registry,
        domain_settings=StubSettings() if harness.registry is not None else None,
        act_versioning_service=harness.act_versioning,
    )
    return harness


async def seed_document(repo: RecordingDocumentRepository, **kwargs) -> Document:
    kwargs.setdefault("filename", "report.pdf")
    kwargs.setdefault("source_path", "uploads/report.pdf")
    return await repo.save(Document(**kwargs))


def outbox_ops(uow) -> list[tuple[OutboxOperation, int]]:
    return [(e.operation, e.aggregate_id) for e in uow.vector_outbox._entries.values()]


def upsert_points(uow) -> list[dict]:
    points = []
    for e in uow.vector_outbox._entries.values():
        if e.operation == OutboxOperation.UPSERT_CHUNKS:
            points.extend(e.payload["points"])
    return points


async def run(harness, doc, **kwargs):
    kwargs.setdefault("storage_key", "uploads/report.pdf")
    kwargs.setdefault("original_filename", doc.filename)
    kwargs.setdefault("visibility", "internal_public")
    kwargs.setdefault("owner_id", 1)
    kwargs.setdefault("group_id", None)
    kwargs.setdefault("replace_id", None)
    kwargs.setdefault("doc_domain", "general")
    await harness.processor.process(document_id=doc.id, **kwargs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_happy_path_indexes_document(tmp_path):
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
    # temp file is removed in _finalize_processing
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_empty_parse_marks_failed(tmp_path):
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
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_parse_failure_marks_failed_and_cleans_temp(tmp_path):
    h = build_harness(tmp_path, parse_error=RuntimeError("boom"))
    doc = await seed_document(h.repo)

    with pytest.raises(RuntimeError, match="boom"):
        await run(h, doc)

    assert h.repo.status_calls[-1]["status"] == "failed"
    assert h.repo.status_calls[-1]["error"] == "boom"
    assert h.metrics.has("inc_documents", "failed")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_status_write_failure_does_not_mask_processing_error(tmp_path):
    """The secondary status-write error is logged while the processing error propagates."""
    h = build_harness(tmp_path, parse_error=RuntimeError("boom"))
    h.repo.fail_failed_status = True
    doc = await seed_document(h.repo)

    with pytest.raises(RuntimeError, match="boom"):
        await run(h, doc)

    assert h.metrics.has("inc_documents", "failed")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_low_quality_pdf_warning_and_metrics(tmp_path):
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
async def test_docx_garbled_warning_without_quality_report(tmp_path):
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
async def test_very_low_char_count_warning(tmp_path):
    docs = [RawDocument(page_content="abc")]
    h = build_harness(tmp_path, docs=docs)
    doc = await seed_document(h.repo, filename="note.txt", source_path="uploads/note.txt")

    await run(h, doc, storage_key="uploads/note.txt", original_filename="note.txt")

    expected = "Документ содержит очень мало текста (3 символов). Возможно, это скан или повреждённый файл."
    assert h.repo.status_calls[-1]["warning"] == expected


@pytest.mark.asyncio
async def test_ambiguous_classification_appends_warning(tmp_path):
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
async def test_legacy_classifier_fallback_when_registry_missing(tmp_path):
    h = build_harness(tmp_path, docs=[RawDocument(page_content="Федеральный закон")])
    doc = await seed_document(h.repo)

    await run(h, doc, doc_domain=None)

    # legacy heuristic: 1 marker hit / 1 KB -> density 1.0 >= threshold 1.0 -> legal
    assert h.repo.domain_calls == [(doc.id, "legal")]
    assert outbox_ops(h.uow) == [(OutboxOperation.UPSERT_CHUNKS, doc.id)]


@pytest.mark.asyncio
async def test_replace_non_versioned_deletes_old_document(tmp_path):
    registry = FakeDomainRegistry(
        profiles={"general": make_profile("general", versioned=False)},
        classify_result=ClassificationResult(
            domain_key="general", confidence=0.9, is_ambiguous=False, is_low_signal=False, candidate_scores={}
        ),
    )
    h = build_harness(tmp_path, registry=registry)
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

    # old doc row is gone (deleted inside conflict resolution)
    assert await h.repo.get_by_id(old.id) is None
    # deferred S3 cleanup runs once, after the transaction
    assert h.storage.deleted == ["uploads/old.pdf"]
    # exactly one DELETE: apply_async_replacement sees the row already gone
    # and skips the duplicate outbox entry
    assert outbox_ops(h.uow) == [
        (OutboxOperation.DELETE_BY_DOCUMENT, old.id),
        (OutboxOperation.UPSERT_CHUNKS, new.id),
    ]


@pytest.mark.asyncio
async def test_replace_versioned_keeps_old_document(tmp_path):
    registry = FakeDomainRegistry(
        profiles={"legal": make_profile("legal", versioned=True)},
        classify_result=ClassificationResult(
            domain_key="legal", confidence=0.9, is_ambiguous=False, is_low_signal=False, candidate_scores={}
        ),
    )
    act_versioning = FakeActVersioningService(
        VersioningResult(
            domain_metadata={"act_number": "42"},
            act_version_id=7,
            act_id=3,
            effective_from=date(2026, 1, 1),
            warning=None,
        )
    )
    h = build_harness(tmp_path, registry=registry, act_versioning=act_versioning)
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

    # versioned domain: previous edition stays as historical
    assert await h.repo.get_by_id(old.id) is not None
    assert h.storage.deleted == []
    assert outbox_ops(h.uow) == [(OutboxOperation.UPSERT_CHUNKS, new.id)]
    # versioning runs exactly once, with the real text (the old empty-text
    # call from the conflict resolver created a garbage act_id=NULL version)
    expected_full_text = "\n".join(d.page_content for d in _default_docs())
    assert [c["full_text"] for c in act_versioning.calls] == [expected_full_text]


@pytest.mark.asyncio
async def test_document_deleted_before_persist_aborts(tmp_path):
    """current_doc is None -> early abort: no chunks, no outbox, status failed."""
    h = build_harness(tmp_path, repo=VanishingDocumentRepository(vanish_id=1, vanish_after=0))
    doc = await seed_document(h.repo)

    await run(h, doc)

    # abort is silent: no FAILED write, the row stays in "processing"
    # (reconciled later by mark_stuck_processing_failed)
    assert h.repo.get_calls == 1
    assert [c["status"] for c in h.repo.status_calls] == ["processing"]
    assert h.repo.domain_calls == []
    assert outbox_ops(h.uow) == []
    assert h.metrics.has("inc_documents", "failed")


@pytest.mark.asyncio
async def test_document_deleted_inside_persist_aborts_with_storage_cleanup(tmp_path):
    """persist sees None -> abort after conflict resolution; S3 cleanup still runs."""
    registry = FakeDomainRegistry(
        profiles={"general": make_profile("general", versioned=False)},
        classify_result=ClassificationResult(
            domain_key="general", confidence=0.9, is_ambiguous=False, is_low_signal=False, candidate_scores={}
        ),
    )
    repo = VanishingDocumentRepository(vanish_id=2, vanish_after=2)
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

    # get(new): conflict resolution + current_doc ok, persist aborted (3x new + 1x old)
    assert repo.get_calls == 4
    assert [c["status"] for c in h.repo.status_calls] == ["processing"]
    assert h.repo.domain_calls == []
    assert outbox_ops(h.uow) == [(OutboxOperation.DELETE_BY_DOCUMENT, old.id)]
    # cleanup runs in finally on every exit path -- the replaced S3 object is
    # not orphaned even though persist aborted
    assert h.storage.deleted == ["uploads/old.pdf"]


@pytest.mark.asyncio
async def test_versioning_warning_and_metadata_enrichment(tmp_path):
    """Join order: quality warning first, versioning warning last."""
    report = PDFQualityReport(total_pages=10, n_ok=5, n_missing=3, n_garbled=2, bad_ratio=0.5)
    registry = FakeDomainRegistry(
        profiles={"legal": make_profile("legal", versioned=True)},
        classify_result=ClassificationResult(
            domain_key="legal", confidence=0.9, is_ambiguous=False, is_low_signal=False, candidate_scores={}
        ),
    )
    act_versioning = FakeActVersioningService(
        VersioningResult(
            domain_metadata={"act_number": "42"},
            act_version_id=7,
            act_id=3,
            effective_from=date(2026, 1, 1),
            warning="Не удалось определить дату вступления в силу",
        )
    )
    h = build_harness(tmp_path, pdf_report=report, registry=registry, act_versioning=act_versioning)
    doc = await seed_document(h.repo, filename="scan.pdf", source_path="uploads/scan.pdf")

    await run(h, doc, storage_key="uploads/scan.pdf", original_filename="scan.pdf", doc_domain=None)

    quality_warning = (
        "Низкое качество распознавания: 3 стр. без текста, 2 стр. с мусорным текстом из 10. "
        "Рекомендуется проверить документ (task pdf:diag) и переиндексировать "
        "после конвертации или ручной вычитки."
    )
    versioning_warning = "Не удалось определить дату вступления в силу"
    final = h.repo.status_calls[-1]
    assert final["warning"] == f"{quality_warning}\n{versioning_warning}"
    assert final["quality_score"] == 0.5

    points = upsert_points(h.uow)
    for p in points:
        assert p["metadata"]["act_version_id"] == 7
        assert p["metadata"]["act_id"] == 3
        assert p["metadata"]["effective_from"] == "2026-01-01"
        assert p["metadata"]["domain_metadata"] == {"act_number": "42"}
    assert act_versioning.calls[0]["document_id"] == doc.id


@pytest.mark.asyncio
async def test_section_prefix_added_to_chunks(tmp_path):
    from infrastructure.ml.langchain_document_parser import LangchainDocumentSplitter

    docs = [RawDocument(page_content="Текст раздела", metadata={"section": "Глава 1"})]
    h = build_harness(tmp_path, docs=docs, splitter=LangchainDocumentSplitter())
    doc = await seed_document(h.repo)

    await run(h, doc)

    points = upsert_points(h.uow)
    assert points[0]["page_content"] == "[Раздел: Глава 1]\nТекст раздела"
    # _attach_metadata_to_docs runs before split -> source lands on every chunk
    assert points[0]["metadata"]["source"] == "report.pdf"
    assert points[0]["metadata"]["doc_date"] == "2026-01-15"


@pytest.mark.asyncio
@pytest.mark.parametrize("warning", [None, "Дата требует ручной проверки"])
async def test_prepared_version_commits_with_chunks_before_cache_invalidation(tmp_path, warning):
    profile = make_profile("legal", versioned=True)
    plan = VersioningPlan([], {"act_number": "42"}, None, date(2026, 1, 1), 0.9, warning)
    service = FakeActVersioningService()
    service.prepare_document_versioning = AsyncMock(return_value=plan)
    events = []
    active_uows = []

    async def create_version(uow, *args):
        assert active_uows == [uow]
        assert uow.chunks._chunks == []
        events.append("version")
        return SimpleNamespace(
            id=7,
            act_id=3,
            effective_from=date(2026, 1, 1),
            effective_to=date(2027, 1, 1),
            is_current=False,
        )

    async def invalidate(*args):
        assert active_uows == []
        assert h.uow._committed
        assert len(upsert_points(h.uow)) == 2
        events.append("invalidate")

    service.create_version_in_uow = AsyncMock(side_effect=create_version)
    service.invalidate_act_answers = AsyncMock(side_effect=invalidate)
    service.invalidate_document_answers = AsyncMock(side_effect=invalidate)
    h = build_harness(
        tmp_path, registry=FakeDomainRegistry(profiles={"legal": profile}), act_versioning=service
    )
    original_create = h.uow_factory.create

    @asynccontextmanager
    async def track_transaction(master=False):
        async with original_create(master=master) as uow:
            active_uows.append(uow)
            try:
                yield uow
            finally:
                active_uows.pop()

    h.uow_factory.create = track_transaction
    doc = await seed_document(h.repo)
    await run(h, doc, doc_domain="legal")

    service.prepare_document_versioning.assert_awaited_once_with(
        profile, "\n".join(d.page_content for d in _default_docs())
    )
    assert service.calls == []  # The legacy versioning path must not run as well.
    points = upsert_points(h.uow)
    assert all(p["metadata"]["domain_metadata"] == {"act_number": "42"} for p in points)
    if warning:
        service.create_version_in_uow.assert_not_awaited()
        service.invalidate_document_answers.assert_awaited_once_with(doc.id)
        assert events == ["invalidate"]
        assert h.repo.status_calls[-1]["warning"] == warning
        assert all("act_version_id" not in p["metadata"] for p in points)
    else:
        assert events == ["version", "invalidate"]
        service.invalidate_act_answers.assert_awaited_once_with(3)
        assert all(p["metadata"]["act_version_id"] == 7 for p in points)
        assert all(p["metadata"]["effective_to"] == "2027-01-01" for p in points)
        assert all(p["metadata"]["is_current"] is False for p in points)


@pytest.mark.asyncio
@pytest.mark.parametrize("registered,prepared", [(False, False), (True, False), (True, True)])
async def test_versioning_without_profile_or_plan_still_indexes(tmp_path, registered, prepared):
    service = FakeActVersioningService()
    if prepared:
        service.prepare_document_versioning = AsyncMock(return_value=None)
    registry = FakeDomainRegistry(profiles={"general": make_profile()} if registered else {})
    h = build_harness(tmp_path, registry=registry, act_versioning=service)
    doc = await seed_document(h.repo)
    await run(h, doc)
    assert h.metrics.has("inc_documents", "indexing")
    assert all("act_version_id" not in p["metadata"] for p in upsert_points(h.uow))
    assert len(service.calls) == int(registered and not prepared)
    if prepared:
        service.prepare_document_versioning.assert_awaited_once()


@pytest.mark.asyncio
async def test_cancelled_versioning_cleans_temp_without_marking_failed(tmp_path):
    service = FakeActVersioningService()
    service.prepare_document_versioning = AsyncMock(side_effect=asyncio.CancelledError)
    h = build_harness(
        tmp_path, registry=FakeDomainRegistry(profiles={"general": make_profile()}), act_versioning=service
    )
    doc = await seed_document(h.repo)
    with pytest.raises(asyncio.CancelledError):
        await run(h, doc)
    assert [c["status"] for c in h.repo.status_calls] == ["processing"]
    assert outbox_ops(h.uow) == []
    assert h.metrics.has("inc_documents", "failed")
    assert h.metrics.has("inc_chunks", 2)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_concurrent_processing_bounds_parser_and_keeps_run_state_separate(tmp_path):
    h = build_harness(tmp_path)
    first = await seed_document(h.repo, filename="first.pdf")
    second = await seed_document(h.repo, filename="second.pdf")
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    downloaded_second = asyncio.Event()
    release = threading.Event()
    original_parse = h.parser.parse
    thread_ids = []

    def blocking_parse(path):
        thread_ids.append(threading.get_ident())
        if len(thread_ids) == 1:
            loop.call_soon_threadsafe(entered.set)
            if not release.wait(timeout=5):
                raise RuntimeError("Parser was not released")
        return original_parse(path)

    original_download = h.storage.download_to_temp

    async def download(key):
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
    assert list(tmp_path.iterdir()) == []
