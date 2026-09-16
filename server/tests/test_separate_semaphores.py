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

    @pytest.mark.asyncio
    async def test_generation_semaphore_permits(self):
        """Generation semaphore allows expected number of concurrent acquires."""
        reg = MLClientRegistry()
        sem = reg.generation_semaphore
        acquired = []
        for _ in range(12):
            await sem.acquire()
            acquired.append(True)
        # 13th acquire should timeout
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(sem.acquire(), timeout=0.05)
        for _ in acquired:
            sem.release()

    @pytest.mark.asyncio
    async def test_auxiliary_semaphore_permits(self):
        """Auxiliary semaphore allows expected number of concurrent acquires."""
        reg = MLClientRegistry()
        sem = reg.auxiliary_semaphore
        acquired = []
        for _ in range(4):
            await sem.acquire()
            acquired.append(True)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(sem.acquire(), timeout=0.05)
        for _ in acquired:
            sem.release()

    @pytest.mark.asyncio
    async def test_embedding_semaphore_permits(self):
        """Embedding semaphore allows expected number of concurrent acquires."""
        reg = MLClientRegistry()
        sem = reg.embedding_semaphore
        acquired = []
        for _ in range(16):
            await sem.acquire()
            acquired.append(True)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(sem.acquire(), timeout=0.05)
        for _ in acquired:
            sem.release()

    @pytest.mark.asyncio
    async def test_ingestion_semaphore_permits(self):
        """Ingestion semaphore allows expected number of concurrent acquires."""
        reg = MLClientRegistry()
        sem = reg.ingestion_semaphore
        acquired = []
        for _ in range(8):
            await sem.acquire()
            acquired.append(True)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(sem.acquire(), timeout=0.05)
        for _ in acquired:
            sem.release()

    def test_semaphores_are_different_objects(self):
        reg = MLClientRegistry()
        assert reg.generation_semaphore is not reg.auxiliary_semaphore
        assert reg.embedding_semaphore is not reg.ingestion_semaphore

    def test_semaphore_same_object_on_repeated_access(self):
        reg = MLClientRegistry()
        assert reg.generation_semaphore is reg.generation_semaphore
        assert reg.auxiliary_semaphore is reg.auxiliary_semaphore

    def test_semaphore_eagerly_created(self):
        """Semaphores are created in __init__, not lazily."""
        reg = MLClientRegistry()
        assert reg.generation_semaphore is not None
        assert reg.auxiliary_semaphore is not None
        assert reg.embedding_semaphore is not None
        assert reg.ingestion_semaphore is not None

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

    @pytest.mark.asyncio
    async def test_ingestion_independent_from_embedding(self):
        """Exhausting ingestion semaphore does not block embedding semaphore."""
        reg = MLClientRegistry()
        ing_sem = reg.ingestion_semaphore
        emb_sem = reg.embedding_semaphore

        # Exhaust ingestion semaphore
        acquired = []
        for _ in range(8):
            await ing_sem.acquire()
            acquired.append(True)

        # Embedding should still be available
        emb_acquired = await asyncio.wait_for(emb_sem.acquire(), timeout=0.1)
        assert emb_acquired is True
        emb_sem.release()

        # Cleanup
        for _ in acquired:
            ing_sem.release()
