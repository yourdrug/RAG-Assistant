"""Carry the authenticated enqueue initiator into worker execution and retries."""

from contextlib import contextmanager
from functools import wraps
from contextvars import ContextVar
from uuid import NAMESPACE_URL, uuid4, uuid5

from application.budget import billing_principal
from domain.runtime.cost import BudgetError


job_operations: ContextVar[tuple[str, dict[str, int]] | None] = ContextVar('job_operations', default=None)


def block_job_dispatch():
    scope = job_operations.get()
    if scope is not None:
        scope[1]['!dispatch-blocked'] = -1


def ensure_job_dispatch_allowed():
    scope = job_operations.get()
    if scope is not None and '!dispatch-blocked' in scope[1]:
        raise BudgetError('Job contains a reserved paid operation; automatic replay is blocked')


def next_operation_id(request_key: str = '') -> str:
    ensure_job_dispatch_allowed()
    scope = job_operations.get()
    if scope is None:
        return str(uuid4())
    job_id, occurrences = scope
    sequence = occurrences.get(request_key, 0)
    occurrences[request_key] = sequence + 1
    # Request keys keep parallel calls stable even when their execution order changes.
    return str(uuid5(NAMESPACE_URL, f'budget:{job_id}:llm:{request_key}:{sequence}'))


@contextmanager
def job_billing_scope(name, ctx, kwargs, cost_principal=None, cost_operation_id=None):
    token = billing_principal.set(cost_principal)
    # Enqueued scope survives retry/restart; fall back for older queued jobs.
    job_id = cost_operation_id or f'{name}:{kwargs.get("job_id", ctx.get("job_id"))}'
    operation_token = job_operations.set((job_id, {}))
    try:
        yield
    finally:
        job_operations.reset(operation_token)
        billing_principal.reset(token)


def bill_job_to_initiator(function):
    @wraps(function)
    async def run(ctx, *args, cost_principal=None, cost_operation_id=None, **kwargs):
        with job_billing_scope(function.__name__, ctx, kwargs, cost_principal, cost_operation_id):
            return await function(ctx, *args, **kwargs)

    return run
