"""Queue producers and worker registrations must agree on task names."""

import asyncio
from unittest.mock import AsyncMock, Mock

from infrastructure.worker import queue
from infrastructure.worker.sweep import run_sweep_task
from infrastructure.worker.checkpoint_cleanup import cron_sweep_checkpoint_cleanup
from presentation.cli.commands import worker as worker_command


def test_enqueued_sweep_resolves_to_registered_handler(monkeypatch):
    pool = Mock(enqueue_job=AsyncMock())
    monkeypatch.setattr(queue, "get_pool", AsyncMock(return_value=pool))
    worker_instance = Mock()
    worker_factory = Mock(return_value=worker_instance)
    monkeypatch.setattr(worker_command, "Worker", worker_factory)

    worker_command.worker()
    asyncio.run(queue.enqueue_sweep(sweep_id=1, job_id=2))

    task_name = pool.enqueue_job.call_args.args[0]
    functions = worker_factory.call_args.kwargs["functions"]
    matching_handlers = [function for function in functions if function.name == task_name]
    assert len(matching_handlers) == 1
    assert matching_handlers[0].coroutine.__wrapped__ is run_sweep_task
    assert pool.enqueue_job.call_args.kwargs["sweep_id"] == 1
    assert pool.enqueue_job.call_args.kwargs["job_id"] == 2
    cleanup_jobs = [
        job
        for job in worker_factory.call_args.kwargs["cron_jobs"]
        if job.coroutine is cron_sweep_checkpoint_cleanup
    ]
    assert len(cleanup_jobs) == 1
    assert cleanup_jobs[0].minute == set(range(60))
    worker_instance.run.assert_called_once()


def test_enqueued_deletion_resolves_to_registered_handler(monkeypatch):
    from infrastructure.worker.document_deletion import delete_document

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
    assert handlers[0].coroutine.__wrapped__ is delete_document
    assert pool.enqueue_job.call_args.kwargs["document_id"] == 7
    assert pool.enqueue_job.call_args.kwargs["user_id"] == 1
    assert pool.enqueue_job.call_args.kwargs["job_id"] == 42


def test_enqueued_billing_scope_survives_worker_retry(monkeypatch):
    from application.ports.billing_context import billing_principal
    from infrastructure.budget.job_context import bill_job_to_initiator, next_operation_id, job_operations

    pool = Mock(enqueue_job=AsyncMock())
    monkeypatch.setattr(queue, 'get_pool', AsyncMock(return_value=pool))
    observed = []

    @bill_job_to_initiator
    async def action(ctx):
        observed.append((billing_principal.get(), next_operation_id(), next_operation_id()))

    async def run():
        token = billing_principal.set('user:123')
        try:
            await queue.enqueue_arq('test', 'action', arq_job_id='job')
        finally:
            billing_principal.reset(token)
        arguments = pool.enqueue_job.call_args.kwargs
        payload = {key: value for key, value in arguments.items() if not key.startswith('_')}
        await action({'job_id': 'job', 'job_try': 1}, **payload)
        await action({'job_id': 'job', 'job_try': 2}, **payload)
        assert observed[0] == observed[1]
        assert observed[0][0] == 'user:123'
        assert observed[0][1] != observed[0][2]
        assert job_operations.get() is None
        assert billing_principal.get() is None
        await queue.enqueue_arq('test', 'action', arq_job_id='job')
        assert pool.enqueue_job.call_args.kwargs['cost_operation_id'] != payload['cost_operation_id']

    asyncio.run(run())
