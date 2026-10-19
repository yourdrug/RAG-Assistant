"""The public ingestion facade shares orchestration across API, CLI and workers."""

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.requests import Request

from application.ingestion import (
    IngestionPort,
    IngestionScope,
    IngestionService,
    IngestRegistryItemDTO,
    IngestRegistryResult,
    S3IngestionTargets,
)
from composition.application import ApplicationContainer
from domain.value_objects.visibility import DocumentVisibility
from infrastructure.worker import tasks as worker_tasks
from presentation.api.dependencies import create_ingest_service, create_ingestion_port
from presentation.cli.commands import ingest


@pytest.fixture
def collaborators() -> SimpleNamespace:
    return SimpleNamespace(batch=AsyncMock(), single=AsyncMock(), registry=AsyncMock(), uploads=AsyncMock())


@pytest.fixture
def service(collaborators: SimpleNamespace) -> IngestionService:
    return IngestionService(
        batch_workflow=collaborators.batch,
        single_file_workflow=collaborators.single,
        registry=collaborators.registry,
        targets=S3IngestionTargets(),
        uploads=collaborators.uploads,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("reset", [False, True])
async def test_full_ingestion_resolves_default_and_preserves_scope(
    service: IngestionService, collaborators: SimpleNamespace, reset: bool
) -> None:
    result = await service.run_full(
        reset=reset, domain="legal", visibility=DocumentVisibility.INTERNAL_GROUP, group_id=7
    )
    collaborators.batch.run.assert_awaited_once_with(
        "docs/", reset=reset, scope=IngestionScope("legal", DocumentVisibility.INTERNAL_GROUP, 7, None)
    )
    assert result.status == "started"
    assert result.docs_dir == "docs/"
    assert result.mode == ("RESET + full reindex" if reset else "APPEND (new files only)")


@pytest.mark.asyncio
@pytest.mark.parametrize("force", [False, True])
async def test_single_ingestion_preserves_scope_and_force_order(
    service: IngestionService, collaborators: SimpleNamespace, force: bool
) -> None:
    events = []
    collaborators.registry.delete.side_effect = lambda filename: events.append(("delete", filename))
    collaborators.single.run.side_effect = lambda path, **_kwargs: events.append(("run", path))
    result = await service.run_single(
        "docs/report.pdf",
        force=force,
        domain="general",
        visibility=DocumentVisibility.CLIENT_PRIVATE,
        client_id=9,
    )
    assert events == ([("delete", "report.pdf")] if force else []) + [("run", "docs/report.pdf")]
    collaborators.single.run.assert_awaited_once_with(
        "docs/report.pdf", scope=IngestionScope("general", DocumentVisibility.CLIENT_PRIVATE, None, 9)
    )
    assert result.file == "docs/report.pdf"
    assert result.force is force


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["run_single", "run_full"])
async def test_invalid_path_is_rejected_before_side_effects(
    service: IngestionService, collaborators: SimpleNamespace, method: str
) -> None:
    options = {"force": True} if method == "run_single" else {}
    with pytest.raises(ValueError, match="S3 key"):
        await getattr(service, method)("docs/../secret.pdf", **options)
    assert collaborators.registry.mock_calls == []
    collaborators.single.run.assert_not_awaited()
    collaborators.batch.run.assert_not_awaited()


@pytest.mark.asyncio
async def test_force_delete_failure_stops_ingestion(
    service: IngestionService, collaborators: SimpleNamespace
) -> None:
    collaborators.registry.delete.side_effect = RuntimeError("registry unavailable")
    with pytest.raises(RuntimeError, match="registry unavailable"):
        await service.run_single("docs/report.pdf", force=True)
    collaborators.single.run.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("method,workflow", [("run_full", "batch"), ("run_single", "single")])
async def test_workflow_failure_propagates(
    service: IngestionService, collaborators: SimpleNamespace, method: str, workflow: str
) -> None:
    getattr(collaborators, workflow).run.side_effect = RuntimeError("index unavailable")
    with pytest.raises(RuntimeError, match="index unavailable"):
        await getattr(service, method)("docs/report.pdf")


@pytest.mark.asyncio
async def test_registry_returns_sorted_dto_with_totals(
    service: IngestionService, collaborators: SimpleNamespace
) -> None:
    indexed_at = datetime(2026, 1, 2, 3, 4, tzinfo=UTC)
    collaborators.registry.list_all.return_value = {
        "z.pdf": {"chunks": 3, "chars": 30, "source": "docs/z.pdf", "indexed_at": indexed_at},
        "a.pdf": {},
    }
    result = await service.get_registry()
    assert result == IngestRegistryResult(
        2,
        3,
        [
            IngestRegistryItemDTO("a.pdf", 0, 0),
            IngestRegistryItemDTO("z.pdf", 3, 30, indexed_at, "docs/z.pdf"),
        ],
    )


def test_api_dependencies_share_one_service(service: IngestionService) -> None:
    application = ApplicationContainer(ingestion_service=service)
    request = Request(
        {
            "type": "http",
            "app": SimpleNamespace(state=SimpleNamespace(container=SimpleNamespace(application=application))),
        }
    )
    assert create_ingest_service(request) is service
    assert create_ingestion_port(request) is service
    assert isinstance(service, IngestionPort)


def test_cli_file_forwards_force_to_shared_operation(monkeypatch: pytest.MonkeyPatch) -> None:
    service = SimpleNamespace(run_single=AsyncMock())
    monkeypatch.setattr(ingest, "create_service", lambda _infra: service)

    async def run(action: Callable[[object], Awaitable[object]]) -> object:
        return await action(MagicMock())

    monkeypatch.setattr(ingest, "run_with_infrastructure", run)
    ingest.ingest_file(
        "docs/report.pdf",
        force=True,
        domain="legal",
        visibility="internal_public",
        group_id=7,
        client_id=9,
    )
    service.run_single.assert_awaited_once_with(
        "docs/report.pdf",
        force=True,
        domain="legal",
        visibility=DocumentVisibility.INTERNAL_PUBLIC,
        group_id=7,
        client_id=9,
    )


def test_cli_registry_formats_shared_dto(monkeypatch: pytest.MonkeyPatch) -> None:
    indexed_at = datetime(2026, 1, 2, 3, 4, tzinfo=UTC)
    service = SimpleNamespace(
        get_registry=AsyncMock(
            return_value=IngestRegistryResult(
                1, 3, [IngestRegistryItemDTO("report.pdf", 3, 3000, indexed_at, "docs/report.pdf")]
            )
        )
    )
    monkeypatch.setattr(ingest, "create_service", lambda _infra: service)

    async def run(action: Callable[[object], Awaitable[object]]) -> object:
        return await action(MagicMock())

    monkeypatch.setattr(ingest, "run_with_infrastructure", run)
    logger = MagicMock()
    monkeypatch.setattr(ingest, "logger", logger)
    ingest.ingest_list()
    logger.info.assert_any_call("Indexed files: %d", 1)
    logger.info.assert_any_call("%-50s  %6s  %8s  %s", "report.pdf", 3, "3,000", "2026-01-02T03:04:00")


@pytest.mark.asyncio
@pytest.mark.parametrize("single", [False, True])
async def test_worker_uses_shared_factory_and_forwards_scope(
    monkeypatch: pytest.MonkeyPatch, single: bool
) -> None:
    service = SimpleNamespace(run_single=AsyncMock(), run_full=AsyncMock())
    factory = MagicMock(return_value=service)
    infra = SimpleNamespace(db=SimpleNamespace(uow_factory=MagicMock()))
    monkeypatch.setattr(worker_tasks, "create_ingestion_service", factory)

    async def tracked(
        uow: object, job_id: int, action: Callable[[], Awaitable[None]], **options: object
    ) -> None:
        assert uow is infra.db.uow_factory
        assert job_id == 42
        assert options["job_try"] == 2
        await action()

    monkeypatch.setattr(worker_tasks, "run_tracked_job", tracked)
    ctx = {"container": SimpleNamespace(infrastructure=infra), "job_try": 2}

    if single:
        await worker_tasks.run_single_ingest(
            ctx,
            resolved="docs/report.pdf",
            domain="general",
            job_id=42,
            visibility="client_private",
            client_id=9,
        )
        service.run_single.assert_awaited_once_with(
            "docs/report.pdf",
            domain="general",
            visibility=DocumentVisibility.CLIENT_PRIVATE,
            group_id=None,
            client_id=9,
        )
    else:
        await worker_tasks.run_full_ingest(
            ctx,
            resolved_dir="docs/",
            reset=True,
            domain="legal",
            job_id=42,
            visibility="internal_group",
            group_id=7,
        )
        service.run_full.assert_awaited_once_with(
            "docs/",
            reset=True,
            domain="legal",
            visibility=DocumentVisibility.INTERNAL_GROUP,
            group_id=7,
            client_id=None,
        )

    factory.assert_called_once_with(infra, uow_factory=infra.db.uow_factory)
