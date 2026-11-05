"""Tests for ActVersioningService — version lifecycle, date trust, chunk sync, outbox."""

from __future__ import annotations

from copy import deepcopy
from datetime import date
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest

from application.act_versioning import VersioningResult
from domain.ingestion.profiles.decree import DecreeDomainProfile
from domain.ingestion.profiles.general import GeneralDomainProfile
from domain.knowledge.enums import DocumentVisibility
from tests.fakes import require_present


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))


from application.act_versioning import ActVersioningService
from domain.ingestion.profiles.types import ReferenceMatch
from domain.knowledge.acts.models import ActVersion, RegulatoryAct
from domain.knowledge.document import Document
from domain.knowledge.index_sync import OutboxOperation
from domain.shared.exceptions import EntityNotFoundError
from tests.fakes import FakeUnitOfWorkFactory


if TYPE_CHECKING:
    from domain.knowledge.index_sync import VectorOutboxEntry


class FakeSettings:
    def __init__(self, threshold: str = "0.85") -> None:
        self._threshold = threshold

    def get(self, key: str, domain_key: str = "") -> str:
        _ = domain_key

        if key == "effective_date_auto_trust_threshold":
            return self._threshold

        return "0"


def decree_profile() -> DecreeDomainProfile:

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
async def test_handle_versioned_upload_matches_existing_act_by_number() -> None:
    service, factory = make_service()
    uow = factory._uow
    existing = await uow.fake_regulatory_acts.save(
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
    assert len(await uow.fake_regulatory_acts.list_all()) == 1


@pytest.mark.asyncio
async def test_handle_versioned_upload_creates_act_for_new_number() -> None:
    service, factory = make_service()
    version = await service.handle_versioned_upload(
        profile=decree_profile(), document_id=1, extracted_refs=act_refs("777")
    )
    assert version.act_id is not None
    acts = await factory._uow.fake_regulatory_acts.list_all()
    assert len(acts) == 1
    assert acts[0].act_number == "777"


@pytest.mark.asyncio
async def test_handle_versioned_upload_without_number_pends_linkage() -> None:
    """No reliable number → act_id=NULL + review queue, never a guessy new act."""
    service, factory = make_service()
    version = await service.handle_versioned_upload(
        profile=decree_profile(), document_id=1, extracted_refs=[]
    )
    assert version.act_id is None
    assert version.date_source == "extracted"
    assert len(await factory._uow.fake_regulatory_acts.list_all()) == 0
    pending = await service.list_pending_review()
    assert version.id in [v.id for v in pending]


@pytest.mark.asyncio
async def test_handle_versioned_upload_unsets_previous_current_and_syncs_chunks() -> None:
    service, factory = make_service()
    uow = factory._uow

    for document_id in (10, 11):
        uow.fake_documents._documents[document_id] = Document(id=document_id)

    act = await uow.fake_regulatory_acts.save(
        RegulatoryAct(id=None, act_type="decree", act_number="5", title="Указ №5", act_date=date(2026, 1, 1))
    )
    old_version = await uow.fake_act_versions.create(
        ActVersion(id=None, act_id=act.id, document_id=10, is_current=True, date_source="manual")
    )
    uow.fake_chunks._chunks.append(
        {"id": 1, "document_id": 10, "act_version_id": old_version.id, "is_current": True}
    )

    new_version = await service.handle_versioned_upload(
        profile=decree_profile(),
        document_id=11,
        extracted_refs=act_refs("5"),
        effective_date=date(2024, 1, 1),
    )

    assert new_version.act_id == act.id
    versions = await uow.fake_act_versions.list_by_act(require_present(act.id))
    current_flags = {v.id: v.is_current for v in versions}
    assert current_flags[old_version.id] is False
    assert current_flags[new_version.id] is True
    # Denormalized chunk flag flipped too
    assert uow.fake_chunks._chunks[0]["is_current"] is False


@pytest.mark.asyncio
async def test_date_source_trusted_only_above_threshold() -> None:
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
async def test_update_version_sets_manual_and_enqueues_qdrant_sync() -> None:
    service, factory = make_service()
    uow = factory._uow
    version = await service.handle_versioned_upload(
        profile=decree_profile(),
        document_id=1,
        extracted_refs=act_refs("9"),
        effective_date=date(2026, 1, 1),
        date_confidence=0.9,
    )
    uow.fake_chunks._chunks.append(
        {"id": 1, "document_id": 1, "act_version_id": version.id, "effective_from": None}
    )

    before_outbox = len(uow.fake_vector_outbox._entries)
    await service.update_version(
        require_present(version.id),
        effective_from=date(2026, 2, 1),
        effective_to=date(2027, 1, 1),
        verified_by=42,
    )

    updated = await uow.fake_act_versions.get_by_id(require_present(version.id))
    assert require_present(updated).date_source == "manual"
    assert require_present(updated).verified_by == 42
    # Postgres chunks synced in the same transaction
    assert uow.fake_chunks._chunks[0]["effective_from"] == date(2026, 2, 1)
    assert uow.fake_chunks._chunks[0]["effective_to"] == date(2027, 1, 1)
    # Qdrant payload sync queued via outbox, without re-embedding
    entries = list(uow.fake_vector_outbox._entries.values())[before_outbox:]
    assert any(e.operation == OutboxOperation.UPDATE_METADATA for e in entries)


@pytest.mark.asyncio
async def test_update_version_unknown_id_raises() -> None:
    service, _ = make_service()
    with pytest.raises(EntityNotFoundError):
        await service.update_version(999, effective_from=date(2026, 1, 1))


# ---------------------------------------------------------------------------
# Unified mechanism: process_document_versioning (API upload AND CLI ingestion)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_document_versioning_full_flow() -> None:
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
    assert require_present(result.domain_metadata)["decree_number"] == "42"
    # Russian month phrase parsed; confidence 0.9 >= 0.85 → trusted date
    # reaches the filterable effective_from field
    assert result.effective_from == date(2026, 6, 1)
    version = await factory._uow.fake_act_versions.get_by_id(result.act_version_id)
    assert require_present(version).date_source == "extracted_trusted"


@pytest.mark.asyncio
async def test_process_document_versioning_low_confidence_date_not_filterable() -> None:
    service, factory = make_service(threshold="0.85")
    profile = decree_profile()
    # Only a header date (confidence 0.3 = signing-date hint) — must NOT reach
    # the filterable effective_from; the version stays a review-queue item
    # without a trusted date
    text = "УКАЗ № 7 от 5 мая 2026 г.\nПОСТАНОВЛЯЮ:\n1. Пункт.\n" * 2
    result = await service.process_document_versioning(profile, document_id=6, full_text=text)

    assert result.effective_from is None
    version = await factory._uow.fake_act_versions.get_by_id(require_present(result.act_version_id))
    assert require_present(version).effective_from is None
    assert require_present(version).date_source == "extracted"
    # Still lands in the manual review queue (no trusted date, no act number)
    pending = await service.list_pending_review()
    assert require_present(version).id in [v.id for v in pending]


@pytest.mark.asyncio
async def test_process_document_versioning_skips_unversioned_profile() -> None:
    service, _ = make_service()

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
async def test_date_correction_rebuilds_chain_chunks_and_outbox(
    new_start: date, expected_ends: list[Any], current_document: int
) -> None:

    service, factory = make_service()
    service._commands._today = lambda: date(2026, 6, 1)
    uow = factory._uow
    versions = []

    for document_id, start in enumerate([date(2024, 1, 1), date(2025, 1, 1), date(2027, 1, 1)], 1):
        uow.fake_documents._documents[document_id] = Document(
            id=document_id, visibility=DocumentVisibility.INTERNAL_PUBLIC
        )
        version = await service.handle_versioned_upload(
            decree_profile(), document_id, act_refs(), effective_date=start
        )
        versions.append(version)
        uow.fake_chunks._chunks.append(
            {
                "act_version_id": version.id,
                "is_current": version.is_current,
                "effective_from": version.effective_from,
                "effective_to": version.effective_to,
            }
        )

    before = len(uow.fake_vector_outbox._entries)
    await service.update_version(require_present(versions[1].id), effective_from=new_start, verified_by=42)
    stored = [await uow.fake_act_versions.get_by_id(require_present(version.id)) for version in versions]
    assert [require_present(version).effective_to for version in stored] == expected_ends
    assert [
        require_present(version).document_id for version in stored if require_present(version).is_current
    ] == [current_document]
    entries = list(uow.fake_vector_outbox._entries.values())[before:]
    assert {entry.aggregate_id for entry in entries} >= {versions[0].id, versions[1].id}

    for stored_version, chunk in zip(stored, uow.fake_chunks._chunks, strict=True):
        assert chunk["effective_from"] == require_present(stored_version).effective_from
        assert chunk["effective_to"] == require_present(stored_version).effective_to
        assert chunk["is_current"] == require_present(stored_version).is_current

    for entry in entries:
        version = require_present(await uow.fake_act_versions.get_by_id(entry.aggregate_id))
        assert entry.operation == OutboxOperation.UPDATE_METADATA
        assert entry.payload["effective_to"] == (
            version.effective_to.isoformat() if version.effective_to else None
        )
        assert entry.payload["is_current"] == version.is_current

    if new_start == date(2026, 1, 1):
        # The reported historical query now resolves v1 instead of a gap.
        as_of = date(2025, 6, 1)
        assert [
            require_present(v).id
            for v in stored
            if require_present(require_present(v).effective_from) <= as_of
            and (
                require_present(v).effective_to is None
                or as_of < require_present(require_present(v).effective_to)
            )
        ] == [versions[0].id]


@pytest.mark.asyncio
async def test_date_correction_preserves_explicit_expiry() -> None:
    service, factory = make_service()
    version = await service.handle_versioned_upload(
        decree_profile(), 1, act_refs(), effective_date=date(2024, 1, 1)
    )
    await service.update_version(require_present(version.id), effective_to=date(2028, 1, 1))
    await service.update_version(require_present(version.id), effective_from=date(2025, 1, 1))
    stored = await factory._uow.fake_act_versions.get_by_id(require_present(version.id))
    assert require_present(stored).effective_to == date(2028, 1, 1)


@pytest.mark.asyncio
async def test_date_correction_outbox_failure_rolls_back_whole_chain(monkeypatch: pytest.MonkeyPatch) -> None:

    service, factory = make_service()
    uow = factory._uow
    versions = []

    for document_id, start in enumerate([date(2024, 1, 1), date(2025, 1, 1)], 1):
        uow.fake_documents._documents[document_id] = Document(
            id=document_id, visibility=DocumentVisibility.INTERNAL_PUBLIC
        )
        version = await service.handle_versioned_upload(
            decree_profile(), document_id, act_refs(), effective_date=start
        )
        versions.append(version)
        uow.fake_chunks._chunks.append(
            {
                "act_version_id": version.id,
                "is_current": version.is_current,
                "effective_from": version.effective_from,
                "effective_to": version.effective_to,
            }
        )

    original = deepcopy(
        (uow.fake_act_versions._versions, uow.fake_chunks._chunks, uow.fake_vector_outbox._entries)
    )
    enqueue = uow.fake_vector_outbox.enqueue
    calls = 0

    async def failing_enqueue(entry: VectorOutboxEntry) -> None:
        nonlocal calls
        calls += 1
        await enqueue(entry)

        if calls == 2:
            msg = "outbox failure"
            raise RuntimeError(msg)

    monkeypatch.setattr(uow.fake_vector_outbox, "enqueue", AsyncMock(side_effect=failing_enqueue))
    with pytest.raises(RuntimeError, match="outbox failure"):
        await service.update_version(require_present(versions[1].id), effective_from=date(2026, 1, 1))
    assert (
        uow.fake_act_versions._versions,
        uow.fake_chunks._chunks,
        uow.fake_vector_outbox._entries,
    ) == original
    assert factory._uow._rolled_back


@pytest.mark.asyncio
@pytest.mark.parametrize("transfer", [False, True])
@pytest.mark.parametrize("outbox_failure", [False, True])
async def test_relink_rebuilds_both_timelines_atomically(  # noqa: C901, PLR0915 - One scenario checks both timelines and rollback atomically.
    transfer: bool, outbox_failure: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, factory = make_service()
    service._commands._today = lambda: date(2026, 6, 1)
    uow = factory._uow
    versions = []

    for document_id, number, start in [
        (1, "123", date(2024, 1, 1)),
        (2, "123", date(2025, 1, 1)),
        (3, "456", date(2024, 6, 1)),
    ]:
        uow.fake_documents._documents[document_id] = Document(id=document_id)
        version = await service.handle_versioned_upload(
            decree_profile(), document_id, act_refs(number), effective_date=start
        )
        versions.append(version)
        uow.fake_chunks._chunks.append(
            {
                "act_version_id": version.id,
                "is_current": version.is_current,
                "effective_from": version.effective_from,
                "effective_to": version.effective_to,
            }
        )

    source, moved, target = versions
    update = uow.fake_act_versions.update

    async def enforce_current_slot(version: ActVersion) -> None:
        if version.act_id is not None and version.is_current:
            others = await uow.fake_act_versions.list_by_act(version.act_id)
            assert not any(item.id != version.id and item.is_current for item in others)

        await update(version)

    monkeypatch.setattr(uow.fake_act_versions, "update", enforce_current_slot)
    original = deepcopy(
        (uow.fake_act_versions._versions, uow.fake_chunks._chunks, uow.fake_vector_outbox._entries)
    )
    before = len(uow.fake_vector_outbox._entries)

    if outbox_failure:
        enqueue = uow.fake_vector_outbox.enqueue
        calls = 0

        async def failing_enqueue(entry: VectorOutboxEntry) -> None:
            nonlocal calls
            calls += 1
            await enqueue(entry)

            if calls == 2:
                msg = "outbox failure"
                raise RuntimeError(msg)

        monkeypatch.setattr(uow.fake_vector_outbox, "enqueue", failing_enqueue)
        with pytest.raises(RuntimeError, match="outbox failure"):
            await service.update_version(
                require_present(moved.id), act_id=target.act_id if transfer else None, act_id_provided=True
            )
        assert (
            uow.fake_act_versions._versions,
            uow.fake_chunks._chunks,
            uow.fake_vector_outbox._entries,
        ) == original
        assert uow._rolled_back
        return

    await service.update_version(
        require_present(moved.id), act_id=target.act_id if transfer else None, act_id_provided=True
    )
    first = await uow.fake_act_versions.get_by_id(require_present(source.id))
    assert require_present(first).is_current
    assert require_present(first).effective_to is None
    stored_moved = await uow.fake_act_versions.get_by_id(require_present(moved.id))
    assert require_present(stored_moved).act_id == (target.act_id if transfer else None)

    if transfer:
        stored_target = await uow.fake_act_versions.get_by_id(require_present(target.id))
        assert not require_present(stored_target).is_current
        assert require_present(stored_target).effective_to == date(2025, 1, 1)

    for chunk in uow.fake_chunks._chunks:
        version = require_present(await uow.fake_act_versions.get_by_id(chunk["act_version_id"]))
        assert chunk["is_current"] == version.is_current
        assert chunk["effective_to"] == version.effective_to

    entries = list(uow.fake_vector_outbox._entries.values())[before:]
    assert {entry.aggregate_id for entry in entries} == (
        {source.id, moved.id, target.id} if transfer else {source.id, moved.id}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit_expiry", [None, date(2026, 1, 1)])
async def test_moving_historical_edition_drops_only_derived_source_end(explicit_expiry: date | None) -> None:
    service, factory = make_service()
    service._commands._today = lambda: date(2026, 6, 1)
    uow = factory._uow
    versions = []

    for document_id, number, start in [
        (1, "123", date(2024, 1, 1)),
        (2, "123", date(2027, 1, 1)),
        (3, "456", date(2023, 1, 1)),
    ]:
        uow.fake_documents._documents[document_id] = Document(id=document_id)
        versions.append(
            await service.handle_versioned_upload(
                decree_profile(), document_id, act_refs(number), effective_date=start
            )
        )

    moved, successor, target = versions

    if explicit_expiry:
        await service.update_version(require_present(moved.id), effective_to=explicit_expiry)

    await service.update_version(require_present(moved.id), act_id=target.act_id)
    stored = await uow.fake_act_versions.get_by_id(require_present(moved.id))
    assert require_present(stored).effective_to == explicit_expiry
    remaining = await uow.fake_act_versions.get_by_id(require_present(successor.id))
    assert not require_present(remaining).is_current  # Future edition must not be promoted.


@pytest.mark.asyncio
async def test_relinking_expired_version_to_empty_act_never_marks_it_current() -> None:
    service, factory = make_service()
    uow = factory._uow
    target = await uow.fake_regulatory_acts.save(RegulatoryAct(None, "decree", "expired-target", "Target"))
    version = await uow.fake_act_versions.create(
        ActVersion(None, None, 1, date(2024, 1, 1), date(2025, 1, 1), False)
    )
    uow.fake_documents._documents[1] = Document(id=1)
    await service.update_version(require_present(version.id), act_id=target.id)
    stored = await uow.fake_act_versions.get_by_id(require_present(version.id))
    assert require_present(stored).effective_to == date(2025, 1, 1)
    assert not require_present(stored).is_current
