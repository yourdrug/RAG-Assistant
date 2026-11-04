"""Exercise CLI parsing and composition lifecycle boundaries without external services."""

from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast


if TYPE_CHECKING:
    from composition.container import Container
    from infrastructure.database.database import DatabaseManager
    from infrastructure.redis.redis_client import RedisClient
    from infrastructure.scheduling.scheduler import Scheduler

from unittest.mock import AsyncMock

import pytest
from typer.testing import CliRunner

from composition.cli.cli import cli
from composition.cli.commands import config, ingest
from domain.knowledge.enums import DocumentVisibility
import main


@pytest.mark.parametrize(
    "command",
    [
        [],
        ["runserver"],
        ["worker"],
        ["ingest", "run"],
        ["ingest", "file"],
        ["ingest", "upload"],
        ["ingest", "list"],
        ["reconcile", "run"],
        ["config", "resync"],
        ["config", "migrate-env"],
        ["benchmark", "export"],
        ["benchmark", "import-legacy"],
    ],
)
def test_all_cli_commands_can_render_help(command: list[str]) -> None:
    result = CliRunner().invoke(cli.cli, [*command, "--help"])

    assert result.exit_code == 0, result.output
    assert "Usage:" in result.output


@pytest.mark.parametrize("single", [False, True])
def test_cli_parses_ingestion_flags_and_preserves_scope(
    monkeypatch: pytest.MonkeyPatch, single: bool
) -> None:
    service = SimpleNamespace(run_full=AsyncMock(), run_single=AsyncMock())

    async def run(action: Callable[[object], Awaitable[None]]) -> None:
        await action(object())

    monkeypatch.setattr(ingest, "run_with_infrastructure", run)
    monkeypatch.setattr(ingest, "create_service", lambda _infra: service)
    command = (
        ["ingest", "file", "docs/report.pdf", "--force"]
        if single
        else ["ingest", "run", "--docs-dir", "docs/selected/", "--reset"]
    )
    result = CliRunner().invoke(
        cli.cli,
        [*command, "--domain", "legal", "--visibility", "internal_group", "--group-id", "17"],
    )

    assert result.exit_code == 0, result.output
    scope = {
        "domain": "legal",
        "visibility": DocumentVisibility.INTERNAL_GROUP,
        "group_id": 17,
        "client_id": None,
    }

    if single:
        service.run_single.assert_awaited_once_with("docs/report.pdf", force=True, **scope)
        service.run_full.assert_not_awaited()
    else:
        service.run_full.assert_awaited_once_with(docs_dir="docs/selected/", reset=True, **scope)
        service.run_single.assert_not_awaited()


def test_config_failure_logs_type_without_provider_secrets(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(
        config, "run_config_resync", AsyncMock(side_effect=RuntimeError("password=private-value"))
    )
    result = CliRunner().invoke(cli.cli, ["config", "resync"])

    assert result.exit_code == 1
    assert "RuntimeError" in caplog.text
    assert "private-value" not in caplog.text + result.output


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["scheduler", "container", "queue", "database", "redis"])
async def test_shutdown_releases_remaining_resources_after_each_failure(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    events: list[str] = []

    def close(label: str) -> Callable[[], Awaitable[None]]:
        async def action() -> None:
            events.append(label)

            if label == failure:
                message = "cleanup failed"
                raise RuntimeError(message)

        return action

    monkeypatch.setattr(main, "close_arq_pool", close("queue"))
    await main.shutdown_resources(
        cast("Container", SimpleNamespace(dispose=close("container"))),
        cast("Scheduler", SimpleNamespace(shutdown=close("scheduler"))),
        cast("DatabaseManager", SimpleNamespace(disconnect=close("database"))),
        cast("RedisClient", SimpleNamespace(aclose=close("redis"))),
    )

    assert events == ["scheduler", "container", "queue", "database", "redis"]
