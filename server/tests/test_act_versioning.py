"""Tests for ActVersioningService — version lifecycle, date trust, chunk sync, outbox."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import pytest

from application.services.act_versioning_service import ActVersioningService
from domain.domain_profile.protocol import ReferenceMatch
from domain.entities.act_version import ActVersion
from domain.entities.document import Document
from domain.entities.regulatory_act import RegulatoryAct
from domain.entities.vector_outbox_entry import OutboxOperation
from domain.exceptions import EntityNotFound
from fakes import FakeUnitOfWorkFactory


class FakeSettings:
    def __init__(self, threshold: str = "0.85") -> None:
        self._threshold = threshold

    def get(self, key: str, domain_key: str = "") -> str:
        if key == "effective_date_auto_trust_threshold":
            return self._threshold
        return "0"


def decree_profile():
    from domain.domain_profile.profiles.decree import DecreeDomainProfile

    return DecreeDomainProfile(settings=FakeSettings())


def make_service(threshold: str = "0.85") -> tuple[ActVersioningService, FakeUnitOfWorkFactory]:
    factory = FakeUnitOfWorkFactory()
    return ActVersioningService(uow_factory=factory, settings=FakeSettings(threshold)), factory


def act_refs(number: str = "123") -> list[ReferenceMatch]:
    return [
        ReferenceMatch("decree_number", number),
        ReferenceMatch("decree_date", "1 января 2026 г."),
    ]


@pytest.mark.asyncio
async def test_handle_versioned_upload_matches_existing_act_by_number():
    service, factory = make_service()
    uow = factory._uow
    existing = await uow.regulatory_acts.save(
        RegulatoryAct(
            id=None, act_type="decree", act_number="123", title="Указ №123", act_date=date(2026, 1, 1)
        )
    )

    version = await service.handle_versioned_upload(
        profile=decree_profile(), document_id=1, extracted_refs=act_refs("123")
    )

    assert version.act_id == existing.id
    assert version.is_current is True
    # No duplicate act created
    assert len(await uow.regulatory_acts.list_all()) == 1


@pytest.mark.asyncio
async def test_handle_versioned_upload_creates_act_for_new_number():
    service, factory = make_service()
    version = await service.handle_versioned_upload(
        profile=decree_profile(), document_id=1, extracted_refs=act_refs("777")
    )
    assert version.act_id is not None
    acts = await factory._uow.regulatory_acts.list_all()
    assert len(acts) == 1
    assert acts[0].act_number == "777"


@pytest.mark.asyncio
async def test_handle_versioned_upload_without_number_pends_linkage():
    """No reliable number → act_id=NULL + review queue, never a guessy new act."""
    service, factory = make_service()
    version = await service.handle_versioned_upload(
        profile=decree_profile(), document_id=1, extracted_refs=[]
    )
    assert version.act_id is None
    assert version.date_source == "extracted"
    assert len(await factory._uow.regulatory_acts.list_all()) == 0
    pending = await service.list_pending_review()
    assert version.id in [v.id for v in pending]


@pytest.mark.asyncio
async def test_handle_versioned_upload_unsets_previous_current_and_syncs_chunks():
    service, factory = make_service()
    uow = factory._uow
    for document_id in (10, 11):
        uow.documents._documents[document_id] = Document(id=document_id)
    act = await uow.regulatory_acts.save(
        RegulatoryAct(id=None, act_type="decree", act_number="5", title="Указ №5", act_date=date(2026, 1, 1))
    )
    old_version = await uow.act_versions.create(
        ActVersion(id=None, act_id=act.id, document_id=10, is_current=True, date_source="manual")
    )
    uow.chunks._chunks.append(
        {"id": 1, "document_id": 10, "act_version_id": old_version.id, "is_current": True}
    )

    new_version = await service.handle_versioned_upload(
        profile=decree_profile(),
        document_id=11,
        extracted_refs=act_refs("5"),
        effective_date=date(2024, 1, 1),
    )

    assert new_version.act_id == act.id
    versions = await uow.act_versions.list_by_act(act.id)
    current_flags = {v.id: v.is_current for v in versions}
    assert current_flags[old_version.id] is False
    assert current_flags[new_version.id] is True
    # Denormalized chunk flag flipped too
    assert uow.chunks._chunks[0]["is_current"] is False


@pytest.mark.asyncio
async def test_date_source_trusted_only_above_threshold():
    service, _ = make_service(threshold="0.85")
    high = await service.handle_versioned_upload(
        profile=decree_profile(),
        document_id=1,
        extracted_refs=act_refs("1"),
        effective_date=date(2026, 1, 1),
        date_confidence=0.9,
    )
    low = await service.handle_versioned_upload(
        profile=decree_profile(),
        document_id=2,
        extracted_refs=act_refs("2"),
        effective_date=date(2026, 1, 1),
        date_confidence=0.3,
    )
    assert high.date_source == "extracted_trusted"
    assert low.date_source == "extracted"


@pytest.mark.asyncio
async def test_update_version_sets_manual_and_enqueues_qdrant_sync():
    service, factory = make_service()
    uow = factory._uow
    version = await service.handle_versioned_upload(
        profile=decree_profile(),
        document_id=1,
        extracted_refs=act_refs("9"),
        effective_date=date(2026, 1, 1),
        date_confidence=0.9,
    )
    uow.chunks._chunks.append(
        {"id": 1, "document_id": 1, "act_version_id": version.id, "effective_from": None}
    )

    before_outbox = len(uow.vector_outbox._entries)
    await service.update_version(
        version.id,
        effective_from=date(2026, 2, 1),
        effective_to=date(2027, 1, 1),
        verified_by=42,
    )

    updated = await uow.act_versions.get_by_id(version.id)
    assert updated.date_source == "manual"
    assert updated.verified_by == 42
    # Postgres chunks synced in the same transaction
    assert uow.chunks._chunks[0]["effective_from"] == date(2026, 2, 1)
    assert uow.chunks._chunks[0]["effective_to"] == date(2027, 1, 1)
    # Qdrant payload sync queued via outbox, without re-embedding
    entries = list(uow.vector_outbox._entries.values())[before_outbox:]
    assert any(e.operation == OutboxOperation.UPDATE_METADATA for e in entries)


@pytest.mark.asyncio
async def test_update_version_unknown_id_raises():
    service, _ = make_service()
    with pytest.raises(EntityNotFound):
        await service.update_version(999, effective_from=date(2026, 1, 1))


# ---------------------------------------------------------------------------
# Unified mechanism: process_document_versioning (API upload AND CLI ingestion)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_document_versioning_full_flow():
    service, factory = make_service(threshold="0.85")
    profile = decree_profile()
    text = (
        "УКАЗ ПРЕЗИДЕНТА № 42 от 1 марта 2026 г.\n"
        "О некоторых мерах по регулированию экономических отношений\n\n"
        "ПОСТАНОВЛЯЮ:\n"
        "1. Настоящий указ вступает в силу с 1 июня 2026 года.\n"
        "2. Контроль возложить на Министерство финансов.\n"
    )
    result = await service.process_document_versioning(profile, document_id=5, full_text=text)

    assert result.warning is None
    assert result.act_version_id is not None
    assert result.act_id is not None
    assert result.domain_metadata["decree_number"] == "42"
    # Russian month phrase parsed; confidence 0.9 >= 0.85 → trusted date
    # reaches the filterable effective_from field
    assert result.effective_from == date(2026, 6, 1)
    version = await factory._uow.act_versions.get_by_id(result.act_version_id)
    assert version.date_source == "extracted_trusted"


@pytest.mark.asyncio
async def test_process_document_versioning_low_confidence_date_not_filterable():
    service, factory = make_service(threshold="0.85")
    profile = decree_profile()
    # Only a header date (confidence 0.3 = signing-date hint) — must NOT reach
    # the filterable effective_from; the version stays a review-queue item
    # without a trusted date
    text = "УКАЗ № 7 от 5 мая 2026 г.\nПОСТАНОВЛЯЮ:\n1. Пункт.\n" * 2
    result = await service.process_document_versioning(profile, document_id=6, full_text=text)

    assert result.effective_from is None
    version = await factory._uow.act_versions.get_by_id(result.act_version_id)
    assert version.effective_from is None
    assert version.date_source == "extracted"
    # Still lands in the manual review queue (no trusted date, no act number)
    pending = await service.list_pending_review()
    assert version.id in [v.id for v in pending]


@pytest.mark.asyncio
async def test_process_document_versioning_skips_unversioned_profile():
    service, _ = make_service()
    from application.dto.versioning_dto import VersioningResult
    from domain.domain_profile.profiles.general import GeneralDomainProfile

    result = await service.process_document_versioning(
        GeneralDomainProfile(), document_id=1, full_text="text"
    )
    assert result == VersioningResult(
        domain_metadata=None,
        act_version_id=None,
        act_id=None,
        effective_from=None,
        warning=None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "new_start, expected_ends, current_document",
    [
        (date(2026, 1, 1), [date(2026, 1, 1), date(2027, 1, 1), None], 2),
        (date(2023, 1, 1), [date(2027, 1, 1), date(2024, 1, 1), None], 1),
        (date(2028, 1, 1), [date(2027, 1, 1), None, date(2028, 1, 1)], 1),
        (date(2027, 1, 1), [date(2027, 1, 1), date(2027, 1, 1), None], 1),
    ],
)
async def test_date_correction_rebuilds_chain_chunks_and_outbox(new_start, expected_ends, current_document):
    from domain.value_objects.visibility import DocumentVisibility

    service, factory = make_service()
    service._commands._today = lambda: date(2026, 6, 1)
    uow = factory._uow
    versions = []
    for document_id, start in enumerate([date(2024, 1, 1), date(2025, 1, 1), date(2027, 1, 1)], 1):
        uow.documents._documents[document_id] = Document(
            id=document_id, visibility=DocumentVisibility.INTERNAL_PUBLIC
        )
        version = await service.handle_versioned_upload(
            decree_profile(), document_id, act_refs(), effective_date=start
        )
        versions.append(version)
        uow.chunks._chunks.append(
            {
                "act_version_id": version.id,
                "is_current": version.is_current,
                "effective_from": version.effective_from,
                "effective_to": version.effective_to,
            }
        )
    before = len(uow.vector_outbox._entries)
    await service.update_version(versions[1].id, effective_from=new_start, verified_by=42)
    stored = [await uow.act_versions.get_by_id(version.id) for version in versions]
    assert [version.effective_to for version in stored] == expected_ends
    assert [version.document_id for version in stored if version.is_current] == [current_document]
    entries = list(uow.vector_outbox._entries.values())[before:]
    assert {entry.aggregate_id for entry in entries} >= {versions[0].id, versions[1].id}
    for version, chunk in zip(stored, uow.chunks._chunks, strict=True):
        assert chunk["effective_from"] == version.effective_from
        assert chunk["effective_to"] == version.effective_to
        assert chunk["is_current"] == version.is_current
    for entry in entries:
        version = await uow.act_versions.get_by_id(entry.aggregate_id)
        assert entry.operation == OutboxOperation.UPDATE_METADATA
        assert entry.payload["effective_to"] == (
            version.effective_to.isoformat() if version.effective_to else None
        )
        assert entry.payload["is_current"] == version.is_current
    if new_start == date(2026, 1, 1):
        # The reported historical query now resolves v1 instead of a gap.
        as_of = date(2025, 6, 1)
        assert [
            v.id
            for v in stored
            if v.effective_from <= as_of and (v.effective_to is None or as_of < v.effective_to)
        ] == [versions[0].id]


@pytest.mark.asyncio
async def test_date_correction_preserves_explicit_expiry():
    service, factory = make_service()
    version = await service.handle_versioned_upload(
        decree_profile(), 1, act_refs(), effective_date=date(2024, 1, 1)
    )
    await service.update_version(version.id, effective_to=date(2028, 1, 1))
    await service.update_version(version.id, effective_from=date(2025, 1, 1))
    stored = await factory._uow.act_versions.get_by_id(version.id)
    assert stored.effective_to == date(2028, 1, 1)


@pytest.mark.asyncio
async def test_date_correction_outbox_failure_rolls_back_whole_chain():
    from copy import deepcopy
    from unittest.mock import AsyncMock

    from domain.value_objects.visibility import DocumentVisibility

    service, factory = make_service()
    uow = factory._uow
    versions = []
    for document_id, start in enumerate([date(2024, 1, 1), date(2025, 1, 1)], 1):
        uow.documents._documents[document_id] = Document(
            id=document_id, visibility=DocumentVisibility.INTERNAL_PUBLIC
        )
        version = await service.handle_versioned_upload(
            decree_profile(), document_id, act_refs(), effective_date=start
        )
        versions.append(version)
        uow.chunks._chunks.append({"act_version_id": version.id, "is_current": version.is_current})
    original = deepcopy((uow.act_versions._versions, uow.chunks._chunks, uow.vector_outbox._entries))
    enqueue = uow.vector_outbox.enqueue
    calls = 0

    async def failing_enqueue(entry):
        nonlocal calls
        calls += 1
        await enqueue(entry)
        if calls == 2:
            raise RuntimeError("outbox failure")

    uow.vector_outbox.enqueue = AsyncMock(side_effect=failing_enqueue)
    with pytest.raises(RuntimeError, match="outbox failure"):
        await service.update_version(versions[1].id, effective_from=date(2026, 1, 1))
    assert (uow.act_versions._versions, uow.chunks._chunks, uow.vector_outbox._entries) == original
    assert factory._uow._rolled_back
