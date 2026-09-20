"""Tests for LLM circuit breaker — resilience/resilience/circuit_breaker.py."""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import pytest

from infrastructure.resilience.circuit_breaker import (
    CircuitBreakerError,
    LLMCircuitBreaker,
    CBState,
)


@pytest.mark.slow
class TestLLMCircuitBreaker:
    """Core circuit breaker behaviour."""

    @pytest.mark.asyncio
    async def test_call_passes_when_closed(self):
        cb = LLMCircuitBreaker(fail_max=5, timeout_duration=60, operation_name="test")
        mock_fn = AsyncMock(return_value="ok")
        result = await cb.call(mock_fn, 1, 2, key="val")
        assert result == "ok"
        mock_fn.assert_awaited_once_with(1, 2, key="val")
        assert cb.state == "closed"

    @pytest.mark.asyncio
    async def test_opens_after_consecutive_failures(self):
        cb = LLMCircuitBreaker(fail_max=3, timeout_duration=60, operation_name="test")
        failing = AsyncMock(side_effect=RuntimeError("LLM down"))

        # First 2 failures: RuntimeError (breaker still closed)
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await cb.call(failing)

        # 3rd failure: RuntimeError (breaker opens AFTER this call)
        with pytest.raises(RuntimeError):
            await cb.call(failing)

        assert cb.state == "open"

    @pytest.mark.asyncio
    async def test_rejects_when_open(self):
        cb = LLMCircuitBreaker(fail_max=2, timeout_duration=60, operation_name="test")
        failing = AsyncMock(side_effect=RuntimeError("down"))

        # Trip the breaker
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await cb.call(failing)

        # Should be rejected immediately
        ok_fn = AsyncMock(return_value="recovered")
        with pytest.raises(CircuitBreakerError):
            await cb.call(ok_fn)

    @pytest.mark.asyncio
    async def test_success_resets_failure_count(self):
        cb = LLMCircuitBreaker(fail_max=3, timeout_duration=60, operation_name="test")
        failing = AsyncMock(side_effect=RuntimeError("err"))
        ok_fn = AsyncMock(return_value="ok")

        # 2 failures (below threshold)
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await cb.call(failing)

        # 1 success resets the counter
        result = await cb.call(ok_fn)
        assert result == "ok"
        assert cb.state == "closed"

    @pytest.mark.asyncio
    async def test_decorator_usage(self):
        cb = LLMCircuitBreaker(fail_max=5, timeout_duration=60, operation_name="test")

        @cb
        async def my_func(x: int) -> int:
            return x * 2

        result = await my_func(5)
        assert result == 10
        assert my_func.__name__ == "my_func"

    @pytest.mark.asyncio
    async def test_metrics_updated_on_success(self):
        cb = LLMCircuitBreaker(fail_max=5, timeout_duration=60, operation_name="metrics_test")
        ok_fn = AsyncMock(return_value="ok")
        await cb.call(ok_fn)

        from infrastructure.resilience.circuit_breaker import CB_STATE

        assert CB_STATE.labels(operation="metrics_test")._value.get() == 0  # closed

    @pytest.mark.asyncio
    async def test_metrics_updated_on_failure(self):
        cb = LLMCircuitBreaker(fail_max=2, timeout_duration=60, operation_name="metrics_fail")
        failing = AsyncMock(side_effect=RuntimeError("err"))

        # Trip the breaker
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await cb.call(failing)

        assert cb.state == "open"

        # Now rejected calls increment the counter
        from infrastructure.resilience.circuit_breaker import CB_FAILURES

        with pytest.raises(CircuitBreakerError):
            await cb.call(AsyncMock(return_value="nope"))

        assert CB_FAILURES.labels(operation="metrics_fail")._value.get() >= 1

    @pytest.mark.asyncio
    async def test_configurable_fail_max(self):
        cb = LLMCircuitBreaker(fail_max=1, timeout_duration=60, operation_name="cfg_test")
        failing = AsyncMock(side_effect=RuntimeError("err"))

        # 1st failure reaches fail_max=1 → opens
        with pytest.raises(RuntimeError):
            await cb.call(failing)

        assert cb.state == "open"

        # Next call should be rejected
        with pytest.raises(CircuitBreakerError):
            await cb.call(AsyncMock(return_value="nope"))

    @pytest.mark.asyncio
    async def test_half_open_after_timeout(self):
        cb = LLMCircuitBreaker(fail_max=1, timeout_duration=1, operation_name="half_open_test")
        failing = AsyncMock(side_effect=RuntimeError("err"))

        # Trip the breaker
        with pytest.raises(RuntimeError):
            await cb.call(failing)
        assert cb.state == "open"

        # Wait for timeout
        await asyncio.sleep(1.5)

        # Should be half-open now — next success closes it
        ok_fn = AsyncMock(return_value="ok")
        result = await cb.call(ok_fn)
        assert result == "ok"
        assert cb.state == "closed"

    @pytest.mark.asyncio
    async def test_half_open_failure_reopens(self):
        cb = LLMCircuitBreaker(fail_max=1, timeout_duration=1, operation_name="half_open_reopen")
        failing = AsyncMock(side_effect=RuntimeError("err"))

        # Trip the breaker
        with pytest.raises(RuntimeError):
            await cb.call(failing)
        assert cb.state == "open"

        # Wait for timeout
        await asyncio.sleep(1.5)

        # Half-open probe fails → reopen
        with pytest.raises(RuntimeError):
            await cb.call(failing)

        assert cb.state == "open"


class TestLLMCircuitBreakerSplitAPI:
    """Direct tests for check_open, report_success, report_failure split API."""

    def test_check_open_closed_raises_nothing(self):
        cb = LLMCircuitBreaker(fail_max=5, timeout_duration=60, operation_name="split_test")
        cb.check_open()  # Should not raise

    def test_check_open_open_raises_circuit_breaker_error(self):
        cb = LLMCircuitBreaker(fail_max=2, timeout_duration=60, operation_name="split_test")
        cb._state = CBState.OPEN
        cb._opened_at = time.monotonic()  # Freshly opened, timeout not expired
        cb._failure_count = 2

        with pytest.raises(CircuitBreakerError):
            cb.check_open()

    def test_check_open_open_timeout_allows_probe(self):
        cb = LLMCircuitBreaker(fail_max=2, timeout_duration=1, operation_name="split_test")
        cb._state = CBState.OPEN
        cb._opened_at = 0.0
        cb._failure_count = 2

        cb.check_open()
        assert cb._state == CBState.OPEN

    @pytest.mark.asyncio
    async def test_report_success_closes_half_open(self):
        cb = LLMCircuitBreaker(fail_max=2, timeout_duration=1, operation_name="split_test")
        cb._state = CBState.HALF_OPEN
        cb._failure_count = 2

        await cb.report_success()
        assert cb.state == "closed"
        assert cb._failure_count == 0

    @pytest.mark.asyncio
    async def test_report_success_resets_failure_count_in_closed(self):
        cb = LLMCircuitBreaker(fail_max=3, timeout_duration=60, operation_name="split_test")
        cb._failure_count = 2

        await cb.report_success()
        assert cb._failure_count == 0

    @pytest.mark.asyncio
    async def test_report_failure_opens_at_threshold(self):
        cb = LLMCircuitBreaker(fail_max=2, timeout_duration=60, operation_name="split_test")

        await cb.report_failure()
        assert cb.state == "closed"
        assert cb._failure_count == 1

        await cb.report_failure()
        assert cb.state == "open"
        assert cb._failure_count == 2

    @pytest.mark.asyncio
    async def test_report_failure_reopens_half_open_immediately(self):
        cb = LLMCircuitBreaker(fail_max=5, timeout_duration=60, operation_name="split_test")
        cb._state = CBState.HALF_OPEN

        await cb.report_failure()
        assert cb.state == "open"

    @pytest.mark.asyncio
    async def test_concurrent_report_success_and_failure(self):
        cb = LLMCircuitBreaker(fail_max=5, timeout_duration=60, operation_name="split_concurrent")
        cb._state = CBState.HALF_OPEN

        await asyncio.gather(cb.report_success(), cb.report_success())
        assert cb.state == "closed"
