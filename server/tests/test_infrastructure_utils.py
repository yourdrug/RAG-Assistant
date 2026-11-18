"""Infrastructure decorators preserve results, cancellation and singleton ownership."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import logging
import threading
from unittest.mock import Mock

import pytest

from infrastructure.logging.log_buffer import LogBufferHandler
from infrastructure.utils.async_exception_handler import handle_exceptions
from infrastructure.utils.utils import singleton


def test_singleton_constructs_once_across_threads() -> None:
    calls = 0
    barrier = threading.Barrier(8)

    @singleton
    class Service:
        def __init__(self, value: int) -> None:
            nonlocal calls
            calls += 1
            self.value = value

    def create() -> object:
        barrier.wait(timeout=5)
        return Service(42)

    with ThreadPoolExecutor(max_workers=8) as executor:
        instances = list(executor.map(lambda _: create(), range(8)))

    assert calls == 1
    assert all(instance is instances[0] for instance in instances)
    assert Service(99).value == 42


def test_singleton_can_retry_after_constructor_failure() -> None:
    calls = 0

    @singleton
    def create() -> object:
        nonlocal calls
        calls += 1

        if calls == 1:
            message = "construction failed"
            raise RuntimeError(message)

        return object()

    with pytest.raises(RuntimeError, match="construction failed"):
        create()
    instance = create()
    assert create() is instance
    assert calls == 2


@pytest.mark.asyncio
async def test_async_exception_handler_preserves_arguments_and_result() -> None:
    @handle_exceptions
    async def action(number: int, *, suffix: str) -> str:
        return f"{number}{suffix}"

    assert await action(7, suffix="!") == "7!"
    assert action.__name__ == "action"


@pytest.mark.asyncio
async def test_async_exception_handler_logs_failure(caplog: pytest.LogCaptureFixture) -> None:
    @handle_exceptions
    async def action() -> str:
        message = "job failed"
        raise RuntimeError(message)

    assert await action() is None
    assert "Job action failed" in caplog.text
    assert caplog.records[-1].exc_info is not None


@pytest.mark.asyncio
async def test_async_exception_handler_propagates_cancellation() -> None:
    @handle_exceptions
    async def action() -> None:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await action()


def test_log_buffer_routes_formatting_failure_to_error_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    handler = LogBufferHandler()
    failed = Mock()
    monkeypatch.setattr(handler, "handleError", failed)
    record = logging.LogRecord("test", logging.INFO, __file__, 1, "%s %s", ("only-one",), None)
    handler.emit(record)
    failed.assert_called_once_with(record)
    assert handler.get_logs() == []


def test_log_buffer_timestamp_is_parseable_iso_utc() -> None:
    handler = LogBufferHandler()
    record = logging.LogRecord("actions", logging.INFO, __file__, 1, "login | user=%s", (7,), None)
    record.created = 1_700_000_000.5
    handler.emit(record)
    [entry] = handler.get_logs()
    assert entry["timestamp"] == "2023-11-14T22:13:20.500+00:00"
    assert entry["message"] == "login | user=7"
    assert datetime.fromisoformat(entry["timestamp"]).tzinfo is not None
