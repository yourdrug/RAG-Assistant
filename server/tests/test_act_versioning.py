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


def _decree_profile():
    from domain.domain_profile.profiles.decree import DecreeDomainProfile

    return DecreeDomainProfile(settings=FakeSettings())


def _service(threshold: str = "0.85") -> tuple[ActVersioningService, FakeUnitOfWorkFactory]:
    factory = FakeUnitOfWorkFactory()
    return ActVersioningService(uow_factory=factory, settings=FakeSettings(threshold)), factory


def _refs(number: str = "123") -> list[ReferenceMatch]:
    return [
        ReferenceMatch("decree_number", number),
        ReferenceMatch("decree_date", "1 января 2026 г."),
    ]


@pytest.mark.asyncio
async def test_handle_versioned_upload_matches_existing_act_by_number():
    service, factory = _service()
    uow = factory._uow
    existing = await uow.regulatory_acts.save(
        RegulatoryAct(id=None, act_type="decree", act_number="123", title="Указ №123")
    )

    version = await service.handle_versioned_upload(
        profile=_decree_profile(), document_id=1, extracted_refs=_refs("123")
    )

    assert version.act_id == existing.id
    assert version.is_current is True
    # No duplicate act created
    assert len(await uow.regulatory_acts.list_all()) == 1


@pytest.mark.asyncio
async def test_handle_versioned_upload_creates_act_for_new_number():
    service, factory = _service()
    version = await service.handle_versioned_upload(
        profile=_decree_profile(), document_id=1, extracted_refs=_refs("777")
    )
    assert version.act_id is not None
    acts = await factory._uow.regulatory_acts.list_all()
    assert len(acts) == 1
    assert acts[0].act_number == "777"


@pytest.mark.asyncio
async def test_handle_versioned_upload_without_number_pends_linkage():
    """No reliable number → act_id=NULL + review queue, never a guessy new act."""
    service, factory = _service()
    version = await service.handle_versioned_upload(
        profile=_decree_profile(), document_id=1, extracted_refs=[]
    )
    assert version.act_id is None
    assert version.date_source == "extracted"
    assert len(await factory._uow.regulatory_acts.list_all()) == 0
    pending = await service.list_pending_review()
    assert version.id in [v.id for v in pending]


@pytest.mark.asyncio
async def test_handle_versioned_upload_unsets_previous_current_and_syncs_chunks():
    service, factory = _service()
    uow = factory._uow
    act = await uow.regulatory_acts.save(
        RegulatoryAct(id=None, act_type="decree", act_number="5", title="Указ №5")
    )
    old_version = await uow.act_versions.create(
        ActVersion(id=None, act_id=act.id, document_id=10, is_current=True, date_source="manual")
    )
    uow.chunks._chunks.append(
        {"id": 1, "document_id": 10, "act_version_id": old_version.id, "is_current": True}
    )

    new_version = await service.handle_versioned_upload(
        profile=_decree_profile(), document_id=11, extracted_refs=_refs("5")
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
    service, _ = _service(threshold="0.85")
    high = await service.handle_versioned_upload(
        profile=_decree_profile(),
        document_id=1,
        extracted_refs=_refs("1"),
        effective_date=date(2026, 1, 1),
        date_confidence=0.9,
    )
    low = await service.handle_versioned_upload(
        profile=_decree_profile(),
        document_id=2,
        extracted_refs=_refs("2"),
        effective_date=date(2026, 1, 1),
        date_confidence=0.3,
    )
    assert high.date_source == "extracted_trusted"
    assert low.date_source == "extracted"


@pytest.mark.asyncio
async def test_update_version_sets_manual_and_enqueues_qdrant_sync():
    service, factory = _service()
    uow = factory._uow
    version = await service.handle_versioned_upload(
        profile=_decree_profile(),
        document_id=1,
        extracted_refs=_refs("9"),
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
    service, _ = _service()
    with pytest.raises(EntityNotFound):
        await service.update_version(999, effective_from=date(2026, 1, 1))


# ---------------------------------------------------------------------------
# Unified mechanism: process_document_versioning (API upload AND CLI ingestion)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_document_versioning_full_flow():
    service, factory = _service(threshold="0.85")
    profile = _decree_profile()
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
    service, factory = _service(threshold="0.85")
    profile = _decree_profile()
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
    service, _ = _service()
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
