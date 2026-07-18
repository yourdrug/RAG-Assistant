"""
cli.py: CLI-приложение проекта.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from typer import Typer

from cli.commands.benchmark import benchmark_app
from cli.commands.ingest import ingest_app
from cli.commands.pdf_diag import pdf_diag_app
from cli.commands.runserver import runserver


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

        # Индексация документов
        self.cli.add_typer(ingest_app, name="ingest")

        # Бенчмарк
        self.cli.add_typer(benchmark_app, name="benchmark")

        # Диагностика PDF
        self.cli.add_typer(pdf_diag_app, name="pdf-diag")

    def execute_command(self, *args: Any, **kwargs: Any) -> None:
        """Выполнить CLI-команду."""
        self.cli(*args, **kwargs)


cli: CLI = CLI()
