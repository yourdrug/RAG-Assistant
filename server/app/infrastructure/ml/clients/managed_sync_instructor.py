"""Keep synchronous benchmark pools alive until active worker calls finish."""

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
import logging
import threading
from types import SimpleNamespace
from typing import Unpack

from instructor import Instructor
from pydantic import BaseModel

from infrastructure.budget.runtime import BudgetRuntime
from infrastructure.ml.clients.client_lease import active_client_leases
from infrastructure.ml.clients.request_types import CompletionRequest
from infrastructure.utils.async_tasks import finish_owned_task


log = logging.getLogger("default")


class ManagedSyncInstructor:
    def __init__(self, client: Instructor, *, budget_runtime: BudgetRuntime | None = None) -> None:
        self.budget_runtime = budget_runtime
        self.client = client
        self._lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._idle = threading.Event()
        self._idle.set()
        self._active = 0
        self._retired = False
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None

    def __getattr__(self, name: str) -> object:
        return getattr(self.client, name)

    @property
    def chat(self) -> SimpleNamespace:
        return SimpleNamespace(completions=SimpleNamespace(create=self.create))

    @contextmanager
    def lease(self) -> Iterator[None]:
        with self._lock:
            if self._retired and id(self) not in active_client_leases.get():
                message = "Judge client was retired; acquire the current judge client"
                raise RuntimeError(message)

            self._active += 1
            self._idle.clear()
        token = active_client_leases.set(active_client_leases.get() | {id(self)})
        try:
            yield
        finally:
            active_client_leases.reset(token)
            with self._lock:
                self._active -= 1

                if not self._active:
                    self._idle.set()

            if self._retired:
                self.retire()

    def create[T: BaseModel](self, **kwargs: Unpack[CompletionRequest[T]]) -> T:
        with self.lease():
            return self.client.chat.completions.create(**kwargs)

    def close_if_idle(self) -> None:
        with self._lock:
            if not self._retired or self._active or self._closed:
                return

            self._closed = True
        try:
            self.client.client.close()
        except Exception:
            with self._lock:
                self._closed = False
            raise

    def retire(self) -> None:
        with self._lock:
            self._retired = True
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            with self._close_lock:
                self.close_if_idle()
            return

        if self._close_task is None or self._close_task.done():
            self._close_task = loop.create_task(asyncio.to_thread(self.close_when_idle))
            self._close_task.add_done_callback(self.report_close_failure)

    def close_when_idle(self) -> None:
        self._idle.wait()
        with self._close_lock:
            self.close_if_idle()

    @property
    def is_closed(self) -> bool:
        return self._closed and (self._close_task is None or self._close_task.done())

    async def close(self) -> None:
        self.retire()

        if self._close_task is not None:
            try:
                await asyncio.shield(self._close_task)
            except asyncio.CancelledError:
                await finish_owned_task(self._close_task)
                raise

    def report_close_failure(self, task: asyncio.Task[None]) -> None:
        if not task.cancelled() and task.exception() is not None:
            log.error("Failed to close retired benchmark judge pool", exc_info=task.exception())
