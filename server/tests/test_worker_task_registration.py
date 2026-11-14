"""Queue producers and worker registrations must agree on task names."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from application.jobs.principal import job_principal
from composition.cli.commands import worker as worker_command
from infrastructure.worker import queue
from infrastructure.worker.checkpoint_cleanup import cron_sweep_checkpoint_cleanup
from infrastructure.worker.document_deletion import delete_document
from infrastructure.worker.document_dispatch import cron_dispatch_documents
from infrastructure.worker.sweep import run_sweep_task


def test_enqueued_sweep_resolves_to_registered_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = Mock(enqueue_job=AsyncMock())
    monkeypatch.setattr(queue, "get_pool", AsyncMock(return_value=pool))
    worker_instance = Mock()
    worker_factory = Mock(return_value=worker_instance)
    monkeypatch.setattr(worker_command, "Worker", worker_factory)

    worker_command.worker(role=worker_command.WorkerRole.BATCH)
    asyncio.run(queue.enqueue_sweep(sweep_id=1, job_id=2))

    task_name = pool.enqueue_job.call_args.args[0]
    functions = worker_factory.call_args.kwargs["functions"]
    matching_handlers = [function for function in functions if function.name == task_name]
    assert len(matching_handlers) == 1
    assert matching_handlers[0].coroutine is run_sweep_task
    assert pool.enqueue_job.call_args.kwargs["sweep_id"] == 1
    assert pool.enqueue_job.call_args.kwargs["job_id"] == 2
    worker_command.worker(role=worker_command.WorkerRole.MAINTENANCE)
    cleanup_jobs = [
        job
        for job in worker_factory.call_args.kwargs["cron_jobs"]
        if job.coroutine is cron_sweep_checkpoint_cleanup
    ]
    assert len(cleanup_jobs) == 1
    assert cleanup_jobs[0].minute == set(range(60))
    dispatch_jobs = [
        job
        for job in worker_factory.call_args.kwargs["cron_jobs"]
        if job.coroutine is cron_dispatch_documents
    ]
    assert len(dispatch_jobs) == 1
    assert dispatch_jobs[0].second == {0, 30}
    assert dispatch_jobs[0].run_at_startup
    assert worker_instance.run.call_count == 2


def test_enqueued_deletion_resolves_to_registered_handler(monkeypatch: pytest.MonkeyPatch) -> None:

    pool = Mock(enqueue_job=AsyncMock())
    monkeypatch.setattr(queue, "get_pool", AsyncMock(return_value=pool))
    worker_factory = Mock(return_value=Mock())
    monkeypatch.setattr(worker_command, "Worker", worker_factory)
    worker_command.worker()
    asyncio.run(queue.enqueue_document_deletion(document_id=7, user_id=1, job_id=42))
    task_name = pool.enqueue_job.call_args.args[0]
    functions = worker_factory.call_args.kwargs["functions"]
    handlers = [function for function in functions if function.name == task_name]
    assert len(handlers) == 1
    assert handlers[0].coroutine is delete_document
    assert pool.enqueue_job.call_args.kwargs["document_id"] == 7
    assert pool.enqueue_job.call_args.kwargs["user_id"] == 1
    assert pool.enqueue_job.call_args.kwargs["job_id"] == 42


def test_enqueue_carries_the_request_principal(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = Mock(enqueue_job=AsyncMock())
    monkeypatch.setattr(queue, "get_pool", AsyncMock(return_value=pool))

    async def run() -> None:
        token = job_principal.set("user:123")
        try:
            await queue.enqueue_arq("test", "action", arq_job_id="job")
        finally:
            job_principal.reset(token)

    asyncio.run(run())
    arguments = pool.enqueue_job.call_args.kwargs
    assert arguments["cost_principal"] == "user:123"
    assert "cost_operation_id" not in arguments
