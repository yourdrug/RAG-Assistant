"""CLI-приложение проекта."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from typer import Typer

from presentation.cli.commands.backfill_chunk_ids import backfill_app
from presentation.cli.commands.benchmark import benchmark_app
from presentation.cli.commands.config import config_app
from presentation.cli.commands.ingest import ingest_app
from presentation.cli.commands.pdf_diag import pdf_diag_app
from presentation.cli.commands.reconcile import reconcile_app
from presentation.cli.commands.runserver import runserver
from presentation.cli.commands.worker import worker


class CLI:
    """CLI: Точка входа для всех команд."""

    def __init__(self) -> None:
        self.cli: Typer = Typer(help="RAG Assistant — CLI")
        self._register_commands()

    def _register(
        self,
        name: str,
        func: Callable,
        help: str | None = None,
    ) -> None:
        """Зарегистрировать функцию как CLI-команду."""
        self.cli.command(name=name, help=help, add_help_option=True)(func)

    def _register_commands(self) -> None:
        """Зарегистрировать все команды."""
        # Сервер
        self._register("runserver", runserver, help="Запустить uvicorn-сервер")

        # Arq worker
        self._register("worker", worker, help="Запустить Arq worker для фоновых задач")

        # Индексация документов
        self.cli.add_typer(ingest_app, name="ingest")

        # Бенчмарк
        self.cli.add_typer(benchmark_app, name="benchmark")

        # Диагностика PDF
        self.cli.add_typer(pdf_diag_app, name="pdf-diag")

        # Reconcile Qdrant/Postgres
        self.cli.add_typer(reconcile_app, name="reconcile")

        # Backfill chunk IDs
        self.cli.add_typer(backfill_app, name="backfill-chunk-ids")

        # Конфигурация
        self.cli.add_typer(config_app, name="config")

    def execute_command(self, *args: Any, **kwargs: Any) -> None:
        """Выполнить CLI-команду."""
        self.cli(*args, **kwargs)


cli: CLI = CLI()
