"""Tests for Qdrant retry on transient errors."""

from __future__ import annotations

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import pytest

from infrastructure.resilience.retry import retry_on_transient


class TestRetryOnTransient:
    """Verify retry decorator handles transient failures."""

    @pytest.mark.asyncio
    async def test_retry_on_timeout_error(self) -> None:
        call_count = 0

        async def flaky_func() -> str:
            nonlocal call_count
            call_count += 1

            if call_count < 3:
                msg = "Qdrant timeout"
                raise TimeoutError(msg)

            return "ok"

        decorated = retry_on_transient(flaky_func)
        result = await decorated()
        assert result == "ok"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_on_connection_error(self) -> None:
        call_count = 0

        async def flaky_func() -> str:
            nonlocal call_count
            call_count += 1

            if call_count < 2:
                msg = "Connection reset"
                raise ConnectionError(msg)

            return "recovered"

        decorated = retry_on_transient(flaky_func)
        result = await decorated()
        assert result == "recovered"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_no_retry_on_non_transient_error(self) -> None:
        call_count = 0

        async def flaky_func() -> None:
            nonlocal call_count
            call_count += 1
            msg = "Not a transient error"
            raise ValueError(msg)

        decorated = retry_on_transient(flaky_func)
        with pytest.raises(ValueError, match="Not a transient error"):
            await decorated()
        assert call_count == 1  # No retry for ValueError

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self) -> None:
        call_count = 0

        async def always_fails() -> None:
            nonlocal call_count
            call_count += 1
            msg = "Always fails"
            raise TimeoutError(msg)

        decorated = retry_on_transient(always_fails)
        with pytest.raises(TimeoutError):
            await decorated()
        assert call_count == 3  # fail_max=3 attempts

    @pytest.mark.asyncio
    async def test_success_on_first_attempt(self) -> None:
        call_count = 0

        async def ok_func() -> str:
            nonlocal call_count
            call_count += 1
            return "immediately ok"

        decorated = retry_on_transient(ok_func)
        result = await decorated()
        assert result == "immediately ok"
        assert call_count == 1
