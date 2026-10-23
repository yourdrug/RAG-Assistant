"""Keep synchronous benchmark pools alive until active worker calls finish."""

import asyncio
import logging
import threading
from types import SimpleNamespace
from contextlib import contextmanager
from infrastructure.budget.runtime import BudgetRuntime
from infrastructure.ml.clients.client_lease import active_client_leases
from infrastructure.utils.async_tasks import finish_owned_task

log = logging.getLogger("default")


class ManagedSyncInstructor:
    def __init__(self, client, *, budget_runtime: BudgetRuntime | None = None):
        self.budget_runtime = budget_runtime
        self.client = client
        self._lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._idle = threading.Event()
        self._idle.set()
        self._active = 0
        self._retired = False
        self._closed = False
        self._close_task = None

    def __getattr__(self, name):
        return getattr(self.client, name)

    @property
    def chat(self):
        return SimpleNamespace(completions=SimpleNamespace(create=self.create))

    @contextmanager
    def lease(self):
        with self._lock:
            if self._retired and id(self) not in active_client_leases.get():
                raise RuntimeError("Judge client was retired; acquire the current judge client")

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

    def create(self, **kwargs):
        with self.lease():
            return self.client.chat.completions.create(**kwargs)

    def close_if_idle(self):
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

    def retire(self):
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

    def close_when_idle(self):
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

    def report_close_failure(self, task):
        if not task.cancelled() and task.exception() is not None:
            log.error("Failed to close retired benchmark judge pool", exc_info=task.exception())
