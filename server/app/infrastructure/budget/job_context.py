"""Carry the authenticated enqueue initiator into worker execution and retries."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import TYPE_CHECKING, Any
from uuid import NAMESPACE_URL, uuid4, uuid5

from application.budget import billing_principal
from domain.runtime.cost import BudgetError


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Coroutine, Iterator


job_operations: ContextVar[tuple[str, dict[str, int]] | None] = ContextVar("job_operations", default=None)


def block_job_dispatch() -> None:
    scope = job_operations.get()

    if scope is not None:
        scope[1]["!dispatch-blocked"] = -1


def ensure_job_dispatch_allowed() -> None:
    scope = job_operations.get()

    if scope is not None and "!dispatch-blocked" in scope[1]:
        msg = "Job contains a reserved paid operation; automatic replay is blocked"
        raise BudgetError(msg)


def next_operation_id(request_key: str = "") -> str:
    ensure_job_dispatch_allowed()
    scope = job_operations.get()

    if scope is None:
        return str(uuid4())

    job_id, occurrences = scope
    sequence = occurrences.get(request_key, 0)
    occurrences[request_key] = sequence + 1
    # Request keys keep parallel calls stable even when their execution order changes.
    return str(uuid5(NAMESPACE_URL, f"budget:{job_id}:llm:{request_key}:{sequence}"))


@contextmanager
def job_billing_scope(
    name: str,
    ctx: dict[str, Any],
    kwargs: dict[str, object],
    cost_principal: str | None = None,
    cost_operation_id: str | None = None,
) -> Iterator[None]:
    token = billing_principal.set(cost_principal)
    # Enqueued scope survives retry/restart; fall back for older queued jobs.
    job_id = cost_operation_id or f"{name}:{kwargs.get('job_id', ctx.get('job_id'))}"
    operation_token = job_operations.set((job_id, {}))
    try:
        yield
    finally:
        job_operations.reset(operation_token)
        billing_principal.reset(token)


def bill_job_to_initiator[T](function: Callable[..., Awaitable[T]]) -> Callable[..., Coroutine[Any, Any, T]]:
    @wraps(function)
    async def run(
        ctx: dict[str, Any],
        *args: object,
        cost_principal: str | None = None,
        cost_operation_id: str | None = None,
        **kwargs: object,
    ) -> T:
        with job_billing_scope(function.__name__, ctx, kwargs, cost_principal, cost_operation_id):
            return await function(ctx, *args, **kwargs)

    return run
