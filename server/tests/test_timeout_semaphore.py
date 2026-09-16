"""Tests for TimeoutSemaphore behavior — FINDING-002 (CRITICAL)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import pytest

from domain.exceptions.domain_errors import SemaphoreTimeoutError
from infrastructure.ml.clients.client_registry import TimeoutSemaphore


class TestTimeoutSemaphoreAcquire:
    """Acquire succeeds within timeout."""

    @pytest.mark.asyncio
    async def test_acquire_succeeds_when_permits_available(self):
        sem = TimeoutSemaphore(value=2, timeout=1.0)
        result = await sem.acquire()
        assert result is True
        sem.release()

    @pytest.mark.asyncio
    async def test_acquire_succeeds_after_release(self):
        sem = TimeoutSemaphore(value=1, timeout=1.0)
        await sem.acquire()
        sem.release()
        result = await sem.acquire()
        assert result is True
        sem.release()


class TestTimeoutSemaphoreTimeout:
    """Acquire raises SemaphoreTimeoutError when permits are exhausted."""

    @pytest.mark.asyncio
    async def test_acquire_raises_timeout_when_exhausted(self):
        sem = TimeoutSemaphore(value=1, timeout=0.05)
        await sem.acquire()
        with pytest.raises(SemaphoreTimeoutError):
            await sem.acquire()
        sem.release()

    @pytest.mark.asyncio
    async def test_acquire_raises_timeout_with_all_permits_held(self):
        sem = TimeoutSemaphore(value=2, timeout=0.05)
        await sem.acquire()
        await sem.acquire()
        with pytest.raises(SemaphoreTimeoutError):
            await sem.acquire()
        sem.release()
        sem.release()

    @pytest.mark.asyncio
    async def test_concurrent_exhaustion_raises_timeout(self):
        sem = TimeoutSemaphore(value=2, timeout=0.05)
        await sem.acquire()
        await sem.acquire()
        tasks = [sem.acquire() for _ in range(3)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        timeouts = [r for r in results if isinstance(r, SemaphoreTimeoutError)]
        successes = [r for r in results if r is True]
        assert len(successes) == 0
        assert len(timeouts) == 3
        sem.release()
        sem.release()


class TestTimeoutSemaphoreRelease:
    """Release correctly frees permits."""

    @pytest.mark.asyncio
    async def test_release_allows_subsequent_acquire(self):
        sem = TimeoutSemaphore(value=1, timeout=1.0)
        await sem.acquire()
        sem.release()
        result = await sem.acquire()
        assert result is True
        sem.release()

    @pytest.mark.asyncio
    async def test_release_after_timeout_unblocks_waiting(self):
        sem = TimeoutSemaphore(value=1, timeout=2.0)
        await sem.acquire()

        unblocked = False

        async def waiter():
            nonlocal unblocked
            await sem.acquire()
            unblocked = True
            sem.release()

        task = asyncio.create_task(waiter())
        await asyncio.sleep(0.05)
        assert not unblocked
        sem.release()
        await asyncio.wait_for(task, timeout=1.0)
        assert unblocked


class TestTimeoutSemaphoreContextManager:
    """Context manager acquires on enter, releases on exit."""

    @pytest.mark.asyncio
    async def test_context_manager_acquire_and_release(self):
        sem = TimeoutSemaphore(value=1, timeout=1.0)
        async with sem:
            # Permit is held — inner acquire via asyncio.wait_for times out
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(sem.acquire(), timeout=0.05)
        # After context exit, permit is released
        result = await asyncio.wait_for(sem.acquire(), timeout=0.1)
        assert result is True
        sem.release()

    @pytest.mark.asyncio
    async def test_context_manager_releases_on_exception(self):
        sem = TimeoutSemaphore(value=1, timeout=1.0)
        with pytest.raises(ValueError):
            async with sem:
                raise ValueError("boom")
        # Semaphore should be released despite exception
        result = await asyncio.wait_for(sem.acquire(), timeout=0.1)
        assert result is True
        sem.release()

    @pytest.mark.asyncio
    async def test_context_manager_timeout_when_exhausted(self):
        sem = TimeoutSemaphore(value=1, timeout=0.05)
        await sem.acquire()
        with pytest.raises(SemaphoreTimeoutError):
            async with sem:
                pass
        sem.release()


class TestTimeoutSemaphoreEdgeCases:
    """Edge cases and invariants."""

    def test_zero_value_semaphore(self):
        sem = TimeoutSemaphore(value=0, timeout=0.05)
        assert sem._sem._value == 0

    @pytest.mark.asyncio
    async def test_zero_value_always_times_out(self):
        sem = TimeoutSemaphore(value=0, timeout=0.05)
        with pytest.raises(SemaphoreTimeoutError):
            await sem.acquire()

    def test_initial_value_preserved(self):
        sem = TimeoutSemaphore(value=5, timeout=1.0)
        assert sem._sem._value == 5
        assert sem._timeout == 1.0
