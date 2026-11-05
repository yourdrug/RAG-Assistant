"""Tests for the CLI ingestion versioning hook (unified with API upload)."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import AsyncMock, MagicMock

from domain.ingestion.profiles.registry import DomainProfileRegistry


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from typing import TYPE_CHECKING

import pytest

from application.act_versioning import ActVersioningService
from application.ingestion import (
    BatchIngestionWorkflow,
    DocumentSyncService,
    IngestionRegistry,
    IngestionService,
    S3DocumentLoader,
    S3IngestionTargets,
    S3UploadService,
    SingleFileIngestionWorkflow,
)
from domain.ingestion.profiles.decree import DecreeDomainProfile
from domain.ingestion.raw_document import RawDocument as Document
from tests.fakes import FakeUnitOfWorkFactory


if TYPE_CHECKING:
    from application.ingestion.dto import RegistryRecord


class FakeSettings:
    def get(self, key: str, domain_key: str = "") -> str:
        _ = domain_key
        defaults: dict[str, str] = {
            "max_unit_chars": "800",
            "fingerprint_min_points": "1",
            "effective_date_auto_trust_threshold": "0.85",
        }
        return defaults.get(key, "0")


_DECREE_TEXT = (
    "УКАЗ ПРЕЗИДЕНТА № 15 от 10 мая 2026 г.\nПОСТАНОВЛЯЮ:\n"
    "1. Мера первая по совершенствованию системы управления.\n"
    "2. Мера вторая по контролю за исполнением настоящего указа.\n"
)


def _make_service() -> tuple[IngestionService, FakeUnitOfWorkFactory, ActVersioningService]:

    factory = FakeUnitOfWorkFactory()
    versioning = ActVersioningService(uow_factory=factory, settings=FakeSettings())
    vector_store = MagicMock()
    vector_store.set_document_id_by_source = AsyncMock()

    registry = DomainProfileRegistry()
    registry.register(DecreeDomainProfile(settings=FakeSettings()))

    file_storage = MagicMock()
    settings = MagicMock(s3_bucket="test-bucket")
    ingestion_registry = IngestionRegistry(factory, file_storage)
    sync = DocumentSyncService(factory, versioning, registry, FakeSettings())
    loader = S3DocumentLoader(
        file_storage,
        MagicMock(),
        MagicMock(),
        settings,
        domain_registry=registry,
        domain_settings=FakeSettings(),
        registry=ingestion_registry,
    )
    targets = S3IngestionTargets()
    service = IngestionService(
        batch_workflow=BatchIngestionWorkflow(
            vector_store,
            settings,
            loader,
            ingestion_registry,
            sync=sync,
        ),
        single_file_workflow=SingleFileIngestionWorkflow(file_storage, loader, ingestion_registry, sync),
        registry=ingestion_registry,
        targets=targets,
        uploads=S3UploadService(file_storage, targets),
    )
    return service, factory, versioning


@pytest.mark.asyncio
async def test_cli_sync_creates_act_version_for_versioned_domain() -> None:
    service, factory, _versioning = _make_service()
    chunks = [
        Document(
            page_content="1. Мера первая.",
            metadata={"source": "s3://b/docs/ukaz.rtf", "doc_domain": "decree"},
        ),
        Document(
            page_content="2. Мера вторая.",
            metadata={"source": "s3://b/docs/ukaz.rtf", "doc_domain": "decree"},
        ),
    ]
    registry: dict[str, RegistryRecord] = {
        "ukaz.rtf": {"source": "s3://b/docs/ukaz.rtf", "hash": "test", "chars": 200, "chunks": 2}
    }

    await service._batch._sync.sync_documents_to_db(
        registry, {"s3://b/docs/ukaz.rtf": 200}, chunks, {"s3://b/docs/ukaz.rtf": _DECREE_TEXT}
    )

    uow = factory._uow
    versions = list(uow.fake_act_versions._versions.values())
    assert len(versions) == 1
    version = versions[0]
    assert version.act_id is not None
    # The act was found/created by the extracted number
    acts = await uow.fake_regulatory_acts.list_all()
    assert any(a.act_number == "15" for a in acts)
    # Versioning metadata reached the chunks persisted via bulk_insert
    stored = uow.fake_chunks._chunks
    assert stored, "chunks must be persisted"
    assert all(c["act_version_id"] == version.id for c in stored)
    assert all(c["domain_metadata"]["decree_number"] == "15" for c in stored)
    assert version.effective_from is None
    assert all(c["effective_from"] is None for c in stored)


@pytest.mark.asyncio
async def test_cli_sync_skips_versioning_for_general_domain() -> None:
    service, factory, _versioning = _make_service()
    chunks = [
        Document(
            page_content="Обычный текст.", metadata={"source": "s3://b/docs/doc.md", "doc_domain": "general"}
        ),
    ]
    registry: dict[str, RegistryRecord] = {
        "doc.md": {"source": "s3://b/docs/doc.md", "hash": "test", "chars": 20, "chunks": 1}
    }

    await service._batch._sync.sync_documents_to_db(
        registry, {"s3://b/docs/doc.md": 20}, chunks, {"s3://b/docs/doc.md": "Обычный текст."}
    )

    assert len(factory._uow.fake_act_versions._versions) == 0


@pytest.mark.asyncio
async def test_cli_auto_classification_uses_registry() -> None:
    service, _, _ = _make_service()
    text = (
        "УКАЗ ПРЕЗИДЕНТА О МЕРАХ\n"
        "ПОСТАНОВЛЯЮ:\n"
        "1. Пункт первый достаточно длинный для классификации текста документа целиком.\n"
        "2. Пункт второй с контролем за исполнением настоящего указа.\n"
    )
    domain = service._batch._loader.classify_text_domain(text, "auto")
    assert domain == "decree"
