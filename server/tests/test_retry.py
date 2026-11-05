"""Tests for the shared retry_on_transient decorator — resilience/retry.py."""

from __future__ import annotations

from pathlib import Path
import sys

import httpx
from qdrant_client.http.exceptions import ResponseHandlingException


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import pytest

from infrastructure.resilience.retry import retry_on_transient


class TestRetryOnTransient:
    """Verify retry_on_transient retries on the correct exception types."""

    def test_retries_on_timeout_error(self) -> None:
        call_count = 0

        @retry_on_transient
        def flaky() -> str:
            nonlocal call_count
            call_count += 1

            if call_count < 3:
                msg = "timed out"
                raise TimeoutError(msg)

            return "ok"

        result = flaky()
        assert result == "ok"
        assert call_count == 3

    def test_retries_on_connection_error(self) -> None:
        call_count = 0

        @retry_on_transient
        def flaky() -> str:
            nonlocal call_count
            call_count += 1

            if call_count < 2:
                msg = "connection refused"
                raise ConnectionError(msg)

            return "ok"

        result = flaky()
        assert result == "ok"
        assert call_count == 2

    def test_retries_on_qdrant_response_handling_exception(self) -> None:

        call_count = 0

        @retry_on_transient
        def flaky() -> str:
            nonlocal call_count
            call_count += 1

            if call_count < 2:
                msg = "timeout"
                raise ResponseHandlingException(TimeoutError(msg))

            return "ok"

        result = flaky()
        assert result == "ok"
        assert call_count == 2

    def test_retries_on_httpx_transport_error(self) -> None:

        call_count = 0

        @retry_on_transient
        def flaky() -> str:
            nonlocal call_count
            call_count += 1

            if call_count < 2:
                msg = "timeout"
                raise httpx.TimeoutException(msg)

            return "ok"

        result = flaky()
        assert result == "ok"
        assert call_count == 2

    def test_does_not_retry_on_value_error(self) -> None:
        call_count = 0

        @retry_on_transient
        def flaky() -> None:
            nonlocal call_count
            call_count += 1
            msg = "bad input"
            raise ValueError(msg)

        with pytest.raises(ValueError, match="bad input"):
            flaky()
        assert call_count == 1  # no retry

    def test_gives_up_after_max_attempts(self) -> None:
        call_count = 0

        @retry_on_transient
        def always_fails() -> None:
            nonlocal call_count
            call_count += 1
            msg = "always times out"
            raise TimeoutError(msg)

        with pytest.raises(TimeoutError):
            always_fails()
        assert call_count == 3  # stop_after_attempt(3)

    @pytest.mark.asyncio
    async def test_works_with_async_functions(self) -> None:
        call_count = 0

        @retry_on_transient
        async def flaky_async() -> str:
            nonlocal call_count
            call_count += 1

            if call_count < 2:
                msg = "connection refused"
                raise ConnectionError(msg)

            return "async ok"

        result = await flaky_async()
        assert result == "async ok"
        assert call_count == 2
