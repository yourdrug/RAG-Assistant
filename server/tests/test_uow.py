"""Transaction lifecycle and compatibility contracts for the real application UoW."""

from __future__ import annotations

import asyncio
from dataclasses import fields
import importlib
import subprocess
import sys
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from application.ports.base_uow import BaseUnitOfWork as LegacyBaseUnitOfWork
from application.ports.session_protocol import SessionProtocol as LegacySessionProtocol
from application.ports.unit_of_work_factory import UnitOfWorkFactory as LegacyUnitOfWorkFactory
from application.uow import (
    BaseUnitOfWork,
    SessionProtocol,
    UnitOfWork,
    UnitOfWorkFactory,
    UnitOfWorkRepositories,
)
from domain.exceptions import DatabaseError
from infrastructure.uow_factory import UnitOfWorkFactory as SQLAlchemyUnitOfWorkFactory


if TYPE_CHECKING:
    from pathlib import Path


class RecordingSession:
    def __init__(self, *failures: str) -> None:
        self.calls: list[str] = []
        self.failures = failures
        self.error = RuntimeError("session failure")

    async def commit(self) -> None:
        self.calls.append("commit")

        if "commit" in self.failures:
            raise self.error

    async def rollback(self) -> None:
        self.calls.append("rollback")

        if "rollback" in self.failures:
            raise self.error

    async def close(self) -> None:
        self.calls.append("close")

        if "close" in self.failures:
            raise self.error


def make_uow(session: RecordingSession) -> UnitOfWork:
    repositories = UnitOfWorkRepositories(
        **{field.name: MagicMock() for field in fields(UnitOfWorkRepositories)}
    )
    return UnitOfWork(session, repositories)


@pytest.mark.asyncio
async def test_success_commits_after_ordered_event_handlers() -> None:
    session = RecordingSession()
    uow = make_uow(session)
    event = object()

    async def first(received: object) -> None:
        assert received is event
        session.calls.append("first")

    async def second(received: object) -> None:
        assert received is event
        session.calls.append("second")

    uow.on_event(first)
    uow.on_event(second)

    async with uow as entered:
        assert entered is uow
        await uow.publish_event(event)
        assert session.calls == ["first", "second"]

    assert session.calls == ["first", "second", "commit", "close"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failures", [(), ("rollback",), ("close",), ("rollback", "close")])
async def test_body_error_survives_cleanup_failures(
    failures: tuple[str, ...], caplog: pytest.LogCaptureFixture
) -> None:
    session = RecordingSession(*failures)
    error = ValueError("body failure")

    with pytest.raises(ValueError, match="body failure") as raised:
        async with make_uow(session):
            raise error

    assert raised.value is error
    assert session.calls == ["rollback", "close"]
    assert len(caplog.records) == len(failures)
    assert all(record.exc_info is not None for record in caplog.records)


@pytest.mark.asyncio
@pytest.mark.parametrize("failures", [(), ("rollback", "close")])
async def test_task_cancellation_rolls_back_and_closes(failures: tuple[str, ...]) -> None:
    session = RecordingSession(*failures)
    entered = asyncio.Event()

    async def transaction() -> None:
        async with make_uow(session):
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(transaction())
    await entered.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert session.calls == ["rollback", "close"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failures", [("commit",), ("commit", "rollback", "close")])
async def test_commit_failure_is_wrapped_after_cleanup(failures: tuple[str, ...]) -> None:
    session = RecordingSession(*failures)

    with pytest.raises(DatabaseError) as raised:
        async with make_uow(session):
            pass

    assert raised.value.__cause__ is session.error
    assert raised.value.errors == {"detail": "session failure"}
    assert session.calls == ["commit", "rollback", "close"]


@pytest.mark.asyncio
async def test_close_failure_after_commit_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    session = RecordingSession("close")

    async with make_uow(session):
        pass

    assert session.calls == ["commit", "close"]
    assert "Failed to close DB session cleanly" in caplog.text


@pytest.mark.asyncio
async def test_event_handler_failure_rolls_back() -> None:
    session = RecordingSession()
    uow = make_uow(session)
    error = ValueError("handler failure")

    async def fail(event: object) -> None:
        raise error

    uow.on_event(fail)

    with pytest.raises(ValueError, match="handler failure") as raised:
        async with uow:
            await uow.publish_event(object())

    assert raised.value is error
    assert session.calls == ["rollback", "close"]


@pytest.mark.asyncio
@pytest.mark.parametrize("master", [False, True])
async def test_factory_binds_all_repositories_to_selected_session(master: bool) -> None:
    session = MagicMock(spec=RecordingSession)
    database = MagicMock()
    database.get_session.return_value = session
    factory = SQLAlchemyUnitOfWorkFactory(database)

    async with factory.create(master=master) as uow:
        for field in fields(UnitOfWorkRepositories):
            repository = getattr(uow, field.name)

            if field.name == "chunks":
                assert all(part._session is session for part in vars(repository).values())
            else:
                assert any(value is session for value in vars(repository).values())

    database.get_session.assert_called_once_with(master=master)
    session.commit.assert_awaited_once()
    session.close.assert_awaited_once()
    session.rollback.assert_not_awaited()


def test_repository_attributes_keep_supplied_instances() -> None:
    repositories = UnitOfWorkRepositories(
        **{field.name: MagicMock() for field in fields(UnitOfWorkRepositories)}
    )
    uow = UnitOfWork(RecordingSession(), repositories)

    for field in fields(UnitOfWorkRepositories):
        assert getattr(uow, field.name) is getattr(repositories, field.name)


def test_legacy_imports_are_identical_to_canonical_types() -> None:
    assert LegacyBaseUnitOfWork is BaseUnitOfWork
    assert LegacySessionProtocol is SessionProtocol
    assert LegacyUnitOfWorkFactory is UnitOfWorkFactory
    assert importlib.import_module("application.uow.unit_of_work").UnitOfWork is UnitOfWork
    assert isinstance(RecordingSession(), SessionProtocol)
    assert isinstance(SQLAlchemyUnitOfWorkFactory(MagicMock()), UnitOfWorkFactory)


@pytest.mark.parametrize(
    "modules",
    [
        ["application.ports.unit_of_work_factory", "application.uow", "application.ports.base_uow"],
        ["application.uow.unit_of_work", "application.ports.session_protocol", "application.uow.factory"],
        ["application.uow.base", "application.uow.factory", "application.ports.unit_of_work_factory"],
    ],
)
def test_import_orders_in_fresh_interpreter(modules: list[str], tmp_path: Path) -> None:
    script = (
        "import importlib, sys\n"
        f"sys.path[:] = {sys.path!r}\n"
        f"for name in {modules!r}: importlib.import_module(name)\n"
        "from application.ports.unit_of_work_factory import UnitOfWorkFactory as old\n"
        "from application.uow import UnitOfWorkFactory as new\n"
        "assert old is new\n"
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script], cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
