"""Tests for the CLI ingestion versioning hook (unified with API upload)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import pytest
from langchain.schema import Document

from application.services.act_versioning_service import ActVersioningService
from application.services.batch_ingestion import BatchIngestionWorkflow
from application.services.document_loader import S3DocumentLoader
from domain.domain_profile.profiles.decree import DecreeDomainProfile
from fakes import FakeUnitOfWorkFactory
from application.services.ingestion_orchestrator import IngestionService
from application.services.ingestion_registry import IngestionRegistry
from application.services.ingestion_sync import DocumentSyncService
from application.services.ingestion_targets import S3IngestionTargets, S3UploadService
from application.services.single_file_ingestion import SingleFileIngestionWorkflow
from application.services.sparse_index_builder import SparseIndexBuilder


class FakeSettings:
    def get(self, key: str, domain_key: str = "") -> str:
        defaults = {
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


def _make_service():
    from domain.domain_profile.registry import DomainProfileRegistry

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
            sync,
            SparseIndexBuilder(settings, None),
        ),
        single_file_workflow=SingleFileIngestionWorkflow(file_storage, loader, ingestion_registry, sync),
        registry=ingestion_registry,
        targets=targets,
        uploads=S3UploadService(file_storage, targets),
    )
    return service, factory, versioning


@pytest.mark.asyncio
async def test_cli_sync_creates_act_version_for_versioned_domain():
    service, factory, versioning = _make_service()
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
    registry = {"ukaz.rtf": {"source": "s3://b/docs/ukaz.rtf"}}

    await service._batch._sync.sync_documents_to_db(
        registry, {"s3://b/docs/ukaz.rtf": 200}, chunks, {"s3://b/docs/ukaz.rtf": _DECREE_TEXT}
    )

    uow = factory._uow
    versions = list(uow.act_versions._versions.values())
    assert len(versions) == 1
    version = versions[0]
    assert version.act_id is not None
    # The act was found/created by the extracted number
    acts = await uow.regulatory_acts.list_all()
    assert any(a.act_number == "15" for a in acts)
    # Versioning metadata reached the chunks persisted via bulk_insert
    stored = uow.chunks._chunks
    assert stored, "chunks must be persisted"
    assert all(c["act_version_id"] == version.id for c in stored)
    assert all(c["domain_metadata"]["decree_number"] == "15" for c in stored)
    assert version.effective_from is None
    assert all(c["effective_from"] is None for c in stored)


@pytest.mark.asyncio
async def test_cli_sync_skips_versioning_for_general_domain():
    service, factory, versioning = _make_service()
    chunks = [
        Document(
            page_content="Обычный текст.", metadata={"source": "s3://b/docs/doc.md", "doc_domain": "general"}
        ),
    ]
    registry = {"doc.md": {"source": "s3://b/docs/doc.md"}}

    await service._batch._sync.sync_documents_to_db(
        registry, {"s3://b/docs/doc.md": 20}, chunks, {"s3://b/docs/doc.md": "Обычный текст."}
    )

    assert len(factory._uow.act_versions._versions) == 0


@pytest.mark.asyncio
async def test_cli_auto_classification_uses_registry():
    service, _, _ = _make_service()
    text = (
        "УКАЗ ПРЕЗИДЕНТА О МЕРАХ\n"
        "ПОСТАНОВЛЯЮ:\n"
        "1. Пункт первый достаточно длинный для классификации текста документа целиком.\n"
        "2. Пункт второй с контролем за исполнением настоящего указа.\n"
    )
    domain = service._batch._loader.classify_text_domain(text, "auto")
    assert domain == "decree"
