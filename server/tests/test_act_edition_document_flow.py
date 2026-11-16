"""Act editions through document processing: the law pack's hooks around the core pipeline."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import date
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock

import pytest

from application.domain_packs.belarus_law.act_versioning import (
    ActVersioningService,
    VersioningPlan,
    VersioningResult,
)
from application.domain_packs.belarus_law.act_versioning.document_hooks import ActEditionHooks
from domain.domain_packs.belarus_law.acts.models import ActVersion
from domain.ingestion.preview import PDFQualityReport
from domain.ingestion.profiles.registry import ClassificationResult
from domain.knowledge.index_sync import OutboxOperation
from tests.law_pack import LawFakeUnitOfWorkFactory
from tests.test_document_processor_characterization import (
    FakeDomainRegistry,
    _default_docs,
    build_harness,
    make_profile,
    outbox_ops,
    run,
    seed_document,
    upsert_points,
)


if TYPE_CHECKING:
    from pathlib import Path

    from tests.fakes import FakeUnitOfWork


def edition_harness(
    tmp_path: Path, act_versioning: ActVersioningService, registry: Any, **overrides: Any
) -> SimpleNamespace:
    harness = build_harness(
        tmp_path,
        registry=registry,
        uow_factory=LawFakeUnitOfWorkFactory(),
        pack_hooks=ActEditionHooks(act_versioning, registry),
        **overrides,
    )
    harness.act_versioning = act_versioning
    return harness


class FakeActVersioningService(ActVersioningService):
    def __init__(self, result: VersioningResult | None = None, version: ActVersion | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.invalidated_acts: list[int] = []
        self.invalidated_documents: list[int] = []
        self._result = result or VersioningResult()
        self._version = version

    async def prepare_document_versioning(self, profile: Any, full_text: str) -> VersioningPlan | None:
        self.calls.append({"profile": profile, "full_text": full_text})

        if self._result.domain_metadata is None and self._result.warning is None:
            return None

        effective_from = self._version.effective_from if self._version else None
        return VersioningPlan(
            [], self._result.domain_metadata, None, effective_from, None, self._result.warning
        )

    async def create_version_in_uow(
        self, uow: object, profile: object, document_id: int, extracted_refs: object, **kwargs: Any
    ) -> ActVersion:
        _ = uow, profile, extracted_refs, kwargs
        return replace(
            self._version or ActVersion(id=None, act_id=None, document_id=0), document_id=document_id
        )

    async def invalidate_act_answers(self, act_id: int) -> None:
        self.invalidated_acts.append(act_id)

    async def invalidate_document_answers(self, document_id: int) -> None:
        self.invalidated_documents.append(document_id)


@pytest.mark.asyncio
async def test_replace_versioned_keeps_old_document(tmp_path: Path) -> None:
    registry = FakeDomainRegistry(
        profiles={"legal": make_profile("legal", versioned=True)},
        classify_result=ClassificationResult(
            domain_key="legal", confidence=0.9, is_ambiguous=False, is_low_signal=False, candidate_scores={}
        ),
    )
    act_versioning = FakeActVersioningService(
        VersioningResult(domain_metadata={"act_number": "42"}),
        ActVersion(id=7, act_id=3, document_id=0, effective_from=date(2026, 1, 1)),
    )
    h = edition_harness(tmp_path, act_versioning, registry)
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
async def test_versioning_warning_and_metadata_enrichment(tmp_path: Path) -> None:
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
            domain_metadata={"act_number": "42"}, warning="Не удалось определить дату вступления в силу"
        ),
        ActVersion(id=7, act_id=3, document_id=0, effective_from=date(2026, 1, 1)),
    )
    h = edition_harness(tmp_path, act_versioning, registry, pdf_report=report)
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

    # Failed extraction without an existing edition does not attach edition metadata.
    for p in points:
        assert "pack_metadata" not in p["metadata"]
        assert p["metadata"]["domain_metadata"] == {"act_number": "42"}

    assert act_versioning.calls[0]["profile"] is registry.get("legal")


@pytest.mark.asyncio
@pytest.mark.parametrize("warning", [None, "Дата требует ручной проверки"])
async def test_prepared_version_commits_with_chunks_before_cache_invalidation(  # noqa: PLR0915 - One scenario verifies the complete transaction/lifecycle.
    tmp_path: Path, warning: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = make_profile("legal", versioned=True)
    plan = VersioningPlan([], {"act_number": "42"}, None, date(2026, 1, 1), 0.9, warning)
    service = FakeActVersioningService()
    monkeypatch.setattr(service, "prepare_document_versioning", AsyncMock(return_value=plan))
    events = []
    active_uows: list[Any] = []

    async def create_version(
        uow: FakeUnitOfWork, *args: Any, effective_date: Any = None, date_confidence: Any = None
    ) -> SimpleNamespace:
        assert effective_date == plan.effective_from
        assert date_confidence == plan.date_confidence
        assert active_uows == [uow]
        assert uow.fake_chunks._chunks == []
        events.append("version")
        return SimpleNamespace(
            id=7,
            act_id=3,
            effective_from=date(2026, 1, 1),
            effective_to=date(2027, 1, 1),
            is_current=False,
        )

    async def invalidate(*args: Any) -> None:
        assert active_uows == []
        assert h.uow._committed
        assert len(upsert_points(h.uow)) == 2
        events.append("invalidate")

    monkeypatch.setattr(service, "create_version_in_uow", AsyncMock(side_effect=create_version))
    monkeypatch.setattr(service, "invalidate_act_answers", AsyncMock(side_effect=invalidate))
    monkeypatch.setattr(service, "invalidate_document_answers", AsyncMock(side_effect=invalidate))
    h = edition_harness(tmp_path, service, FakeDomainRegistry(profiles={"legal": profile}))
    original_create = h.uow_factory.create

    @asynccontextmanager
    async def track_transaction(master: bool = False) -> Any:
        async with original_create(master=master) as uow:
            active_uows.append(uow)
            try:
                yield uow
            finally:
                active_uows.pop()

    h.uow_factory.create = track_transaction
    doc = await seed_document(h.repo)
    await run(h, doc, doc_domain="legal")

    cast("AsyncMock", service.prepare_document_versioning).assert_awaited_once_with(
        profile, "\n".join(d.page_content for d in _default_docs())
    )
    points = upsert_points(h.uow)
    assert all(p["metadata"]["domain_metadata"] == {"act_number": "42"} for p in points)

    if warning:
        cast("AsyncMock", service.create_version_in_uow).assert_not_awaited()
        cast("AsyncMock", service.invalidate_document_answers).assert_awaited_once_with(doc.id)
        assert events == ["invalidate"]
        assert h.repo.status_calls[-1]["warning"] == warning
        assert all("pack_metadata" not in p["metadata"] for p in points)
    else:
        assert events == ["version", "invalidate"]
        cast("AsyncMock", service.invalidate_act_answers).assert_awaited_once_with(3)
        assert all(p["metadata"]["pack_metadata"]["act_version_id"] == 7 for p in points)
        assert all(p["metadata"]["pack_metadata"]["effective_to"] == "2027-01-01" for p in points)
        assert all(p["metadata"]["pack_metadata"]["is_current"] is False for p in points)


@pytest.mark.asyncio
@pytest.mark.parametrize("registered", [False, True])
async def test_versioning_without_profile_or_plan_still_indexes(
    tmp_path: Path, registered: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = FakeActVersioningService()
    monkeypatch.setattr(service, "prepare_document_versioning", AsyncMock(return_value=None))
    registry = FakeDomainRegistry(profiles={"general": make_profile()} if registered else {})
    h = edition_harness(tmp_path, service, registry)
    doc = await seed_document(h.repo)
    await run(h, doc)
    assert h.metrics.has("inc_documents", "indexing")
    assert all("act_version_id" not in p["metadata"] for p in upsert_points(h.uow))

    if registered:
        cast("AsyncMock", service.prepare_document_versioning).assert_awaited_once()
    else:
        cast("AsyncMock", service.prepare_document_versioning).assert_not_awaited()


@pytest.mark.asyncio
async def test_cancelled_versioning_cleans_temp_without_marking_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = FakeActVersioningService()
    monkeypatch.setattr(service, "prepare_document_versioning", AsyncMock(side_effect=asyncio.CancelledError))
    h = edition_harness(tmp_path, service, FakeDomainRegistry(profiles={"general": make_profile()}))
    doc = await seed_document(h.repo)
    with pytest.raises(asyncio.CancelledError):
        await run(h, doc)
    assert [c["status"] for c in h.repo.status_calls] == ["processing"]
    assert outbox_ops(h.uow) == []
    assert h.metrics.has("inc_documents", "failed")
    assert h.metrics.has("inc_chunks", 2)
    assert list(tmp_path.iterdir()) == []  # noqa: ASYNC240 - Small local fixture I/O in the test setup.
