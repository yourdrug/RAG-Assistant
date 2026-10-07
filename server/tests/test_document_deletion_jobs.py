"""Queued deletion authorizes before enqueue and rechecks permissions in the worker."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from application.services.document_deletion_jobs import DocumentDeletionJobs
from domain.entities.background_job import BackgroundJob
from domain.entities.document import Document
from domain.entities.user import User
from domain.exceptions import BusinessRuleViolation, EntityNotFound
from domain.value_objects.job_status import BackgroundJobStatus
from domain.value_objects.job_type import JobType
from domain.value_objects.roles import UserKind, UserRole
from domain.value_objects.visibility import DocumentVisibility
from fakes import FakeUnitOfWorkFactory
from infrastructure.worker.document_deletion import delete_document


async def harness():
    factory = FakeUnitOfWorkFactory()
    uow = factory._uow
    doc = await uow.documents.save(
        Document(
            filename="report.txt",
            owner_id=1,
            visibility=DocumentVisibility.INTERNAL_PRIVATE,
        )
    )
    uow.background_jobs.create = AsyncMock(
        return_value=BackgroundJob(
            id=42,
            job_type=JobType.DOCUMENT_DELETION,
            related_id=doc.id,
        )
    )
    uow.background_jobs.mark_failed = AsyncMock()
    uow.background_jobs.mark_done = AsyncMock()
    enqueuer = SimpleNamespace(enqueue_document_deletion=AsyncMock())
    return factory, uow, doc, enqueuer, DocumentDeletionJobs(factory, enqueuer)


@pytest.mark.asyncio
async def test_authorized_request_only_queues_and_keeps_document_until_worker_runs():
    factory, uow, doc, enqueuer, service = await harness()
    assert await service.enqueue(doc.id, 1, UserKind.INTERNAL, UserRole.USER) == 42
    assert await uow.documents.get_by_id(doc.id) is doc
    enqueuer.enqueue_document_deletion.assert_awaited_once_with(document_id=doc.id, user_id=1, job_id=42)


@pytest.mark.asyncio
async def test_unowned_document_is_rejected_before_creating_or_queuing_job():
    factory, uow, doc, enqueuer, service = await harness()
    with pytest.raises(BusinessRuleViolation):
        await service.enqueue(doc.id, 2, UserKind.INTERNAL, UserRole.USER)
    uow.background_jobs.create.assert_not_awaited()
    enqueuer.enqueue_document_deletion.assert_not_awaited()


@pytest.mark.asyncio
async def test_enqueue_failure_marks_job_failed_and_preserves_document():
    factory, uow, doc, enqueuer, service = await harness()
    enqueuer.enqueue_document_deletion.side_effect = ConnectionError("Redis down")
    with pytest.raises(ConnectionError):
        await service.enqueue(doc.id, 1, UserKind.INTERNAL, UserRole.USER)
    uow.background_jobs.mark_failed.assert_awaited_once()
    assert await uow.documents.get_by_id(doc.id) is doc


@pytest.mark.asyncio
async def test_job_status_requires_document_ownership_and_matching_job():
    factory, uow, doc, enqueuer, service = await harness()
    job = BackgroundJob(
        id=42, related_id=doc.id, job_type=JobType.DOCUMENT_DELETION, status=BackgroundJobStatus.FAILED
    )
    uow.background_jobs.get_by_id = AsyncMock(return_value=job)
    assert await service.get(doc.id, 42, 1, UserKind.INTERNAL, UserRole.USER) is job
    with pytest.raises(BusinessRuleViolation):
        await service.get(doc.id, 42, 2, UserKind.INTERNAL, UserRole.USER)
    job.related_id = doc.id + 1
    with pytest.raises(EntityNotFound):
        await service.get(doc.id, 42, 1, UserKind.INTERNAL, UserRole.USER)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "active,exists,role",
    [
        (True, True, UserRole.CURATOR),
        (False, True, UserRole.CURATOR),
        (False, False, UserRole.USER),
        (True, True, UserRole.USER),
    ],
)
async def test_worker_rechecks_current_user_and_retry_invalidates_deleted_document(active, exists, role):
    factory, uow, doc, enqueuer, service = await harness()
    uow.users.get_by_id = AsyncMock(return_value=User(id=1, role=role, is_active=active))
    if not exists:
        await uow.documents.delete(doc.id)
    command = SimpleNamespace(delete_document=AsyncMock())
    cache = SimpleNamespace(invalidate_by_document_ids=AsyncMock())
    ctx = {
        "job_try": 3,
        "container": SimpleNamespace(
            infrastructure=SimpleNamespace(
                db=SimpleNamespace(uow_factory=factory),
                services=SimpleNamespace(cache_invalidator=cache, action_logger=Mock()),
            ),
            application=SimpleNamespace(document_command_service=command),
        ),
    }
    await delete_document(ctx, document_id=doc.id, user_id=1, job_id=42)
    if exists and (not active or role == UserRole.USER):
        command.delete_document.assert_not_awaited()
        uow.background_jobs.mark_failed.assert_awaited_once()
        cache.invalidate_by_document_ids.assert_not_awaited()
    else:
        if exists:
            command.delete_document.assert_awaited_once_with(doc.id, 1, role, UserKind.INTERNAL)
        else:
            command.delete_document.assert_not_awaited()
        cache.invalidate_by_document_ids.assert_awaited_once_with([doc.id], raise_on_error=True)
        uow.background_jobs.mark_done.assert_awaited_once_with(42)


@pytest.mark.asyncio
async def test_delete_route_returns_accepted_job_without_executing_delete():
    from presentation.api.routes.documents import delete_document as route, router

    service = SimpleNamespace(enqueue=AsyncMock(return_value=42))
    response = await route(
        7, SimpleNamespace(id=1, kind=UserKind.INTERNAL, role=UserRole.USER), service, Mock()
    )
    assert response.model_dump() == {
        "status": BackgroundJobStatus.PENDING.value,
        "document_id": 7,
        "job_id": 42,
    }
    endpoint = next(
        route
        for route in router.routes
        if route.path == "/documents/{document_id}" and "DELETE" in route.methods
    )
    assert endpoint.status_code == 202
