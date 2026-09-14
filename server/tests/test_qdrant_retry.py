"""Tests for Qdrant retry on transient errors."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import pytest

from infrastructure.resilience.retry import retry_on_transient


class TestRetryOnTransient:
    """Verify retry decorator handles transient failures."""

    @pytest.mark.asyncio
    async def test_retry_on_timeout_error(self):
        call_count = 0

        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TimeoutError("Qdrant timeout")
            return "ok"

        decorated = retry_on_transient(flaky_func)
        result = await decorated()
        assert result == "ok"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_on_connection_error(self):
        call_count = 0

        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Connection reset")
            return "recovered"

        decorated = retry_on_transient(flaky_func)
        result = await decorated()
        assert result == "recovered"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_no_retry_on_non_transient_error(self):
        call_count = 0

        async def flaky_func():
            nonlocal call_count
            call_count += 1
            raise ValueError("Not a transient error")

        decorated = retry_on_transient(flaky_func)
        with pytest.raises(ValueError):
            await decorated()
        assert call_count == 1  # No retry for ValueError

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        call_count = 0

        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise TimeoutError("Always fails")

        decorated = retry_on_transient(always_fails)
        with pytest.raises(TimeoutError):
            await decorated()
        assert call_count == 3  # fail_max=3 attempts

    @pytest.mark.asyncio
    async def test_success_on_first_attempt(self):
        call_count = 0

        async def ok_func():
            nonlocal call_count
            call_count += 1
            return "immediately ok"

        decorated = retry_on_transient(ok_func)
        result = await decorated()
        assert result == "immediately ok"
        assert call_count == 1
