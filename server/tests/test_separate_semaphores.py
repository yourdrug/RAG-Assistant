"""Tests for separate LLM semaphores in MLClientRegistry."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import pytest

from infrastructure.ml.clients.client_registry import MLClientRegistry


class TestSeparateSemaphores:
    """Verify generation and auxiliary semaphores are independent."""

    def test_generation_semaphore_has_correct_permits(self):
        reg = MLClientRegistry()
        sem = reg.generation_semaphore
        assert sem._sem._value == 12  # settings.llm_generation_max_concurrent

    def test_auxiliary_semaphore_has_correct_permits(self):
        reg = MLClientRegistry()
        sem = reg.auxiliary_semaphore
        assert sem._sem._value == 4  # settings.llm_auxiliary_max_concurrent

    def test_semaphores_are_different_objects(self):
        reg = MLClientRegistry()
        assert reg.generation_semaphore is not reg.auxiliary_semaphore

    def test_generation_semaphore_is_lazily_created(self):
        reg = MLClientRegistry()
        assert reg._generation_semaphore is None
        _ = reg.generation_semaphore
        assert reg._generation_semaphore is not None

    def test_auxiliary_semaphore_is_lazily_created(self):
        reg = MLClientRegistry()
        assert reg._auxiliary_semaphore is None
        _ = reg.auxiliary_semaphore
        assert reg._auxiliary_semaphore is not None

    def test_semaphore_same_object_on_repeated_access(self):
        reg = MLClientRegistry()
        assert reg.generation_semaphore is reg.generation_semaphore
        assert reg.auxiliary_semaphore is reg.auxiliary_semaphore

    @pytest.mark.asyncio
    async def test_independent_concurrency(self):
        """Exhausting auxiliary semaphore does not block generation semaphore."""
        reg = MLClientRegistry()
        gen_sem = reg.generation_semaphore
        aux_sem = reg.auxiliary_semaphore

        # Exhaust auxiliary semaphore
        acquired = []
        for _ in range(4):
            await aux_sem.acquire()
            acquired.append(True)

        # Generation should still be available
        gen_acquired = await asyncio.wait_for(gen_sem.acquire(), timeout=0.1)
        assert gen_acquired is True
        gen_sem.release()

        # Cleanup
        for _ in acquired:
            aux_sem.release()
