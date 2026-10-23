"""Cancellation-safe completion of owned infrastructure work."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable


async def finish_owned_task[T](task: asyncio.Task[T]) -> T:
    """Drain owned work even when the caller is cancelled."""
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue

    return task.result()


async def run_sync_io[**P, T](function: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    """Run blocking work in a thread and drain it before propagating cancellation."""
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError as exc:
        try:
            await finish_owned_task(task)
        finally:
            # A failed thread job must not replace the caller's cancellation.
            raise exc
