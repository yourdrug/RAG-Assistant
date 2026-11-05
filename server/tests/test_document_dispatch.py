"""Document submissions survive HTTP termination and Redis publish failures."""

from unittest.mock import AsyncMock, Mock

import pytest

from domain.runtime.jobs import BackgroundJobStatus
from infrastructure.worker import document_dispatch
from tests.test_document_commands_characterization import harness, seed, upload


@pytest.mark.asyncio
async def test_upload_commits_dispatch_payload_with_replacement_and_acl() -> None:
    h = harness()
    old = await seed(h)
    dto = await upload(h, doc_domain="legal")
    jobs = list(h.uow.background_jobs.jobs.values())
    assert len(jobs) == 1
    job = jobs[0]
    assert job.related_id == dto.id
    assert job.dispatch_payload == {
        "document_id": dto.id,
        "storage_key": dto.storage_key,
        "filename": dto.filename,
        "visibility": "internal_private",
        "owner_id": 1,
        "group_id": None,
        "replace_id": old.id,
        "doc_domain": "legal",
        "principal_id": 1,
        "cost_principal": "user:1",
    }
    assert h.factory.events[-1] == "commit"


@pytest.mark.asyncio
async def test_failure_to_record_job_rolls_back_document(monkeypatch: pytest.MonkeyPatch) -> None:
    h = harness()
    monkeypatch.setattr(h.uow.background_jobs, "create", AsyncMock(side_effect=RuntimeError("DB failure")))
    with pytest.raises(RuntimeError, match="DB failure"):
        await upload(h)
    assert await h.uow.documents.get_by_id(1) is None
    assert h.uow.background_jobs.jobs == {}
    assert h.factory.events[-1] == "rollback"


@pytest.mark.asyncio
async def test_dispatch_retries_after_redis_failure_with_stable_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = harness()
    dto = await upload(h)
    publish = AsyncMock(side_effect=[ConnectionError("Redis unavailable"), Mock(), None])
    monkeypatch.setattr(document_dispatch, "enqueue_arq", publish)
    context = {"uow_factory": h.factory}
    # The original HTTP handler is gone; a later worker can recover the job.
    await document_dispatch.cron_dispatch_documents(context)
    assert len(await h.uow.background_jobs.pending_dispatch()) == 1
    await document_dispatch.cron_dispatch_documents(context)
    # Still pending in the queue: publishing again uses Arq duplicate suppression.
    await document_dispatch.cron_dispatch_documents(context)
    assert publish.await_count == 3
    assert publish.await_args_list[0] == publish.await_args_list[1] == publish.await_args_list[2]
    assert publish.await_args is not None
    payload = publish.await_args.kwargs
    assert payload["document_id"] == dto.id
    assert payload["cost_principal"] == "user:1"
    assert payload["cost_operation_id"] == "document_processing:1"
    assert payload["arq_job_id"] == f"process_doc:{dto.id}:1"
    job = h.uow.background_jobs.jobs[1]
    job.status = BackgroundJobStatus.RUNNING
    await document_dispatch.cron_dispatch_documents(context)
    job.status = BackgroundJobStatus.DONE
    await document_dispatch.cron_dispatch_documents(context)
    assert publish.await_count == 3
