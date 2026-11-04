"""CLI-приложение проекта."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from typer import Typer

from composition.cli.commands.benchmark import benchmark_app
from composition.cli.commands.config import config_app
from composition.cli.commands.ingest import ingest_app
from composition.cli.commands.reconcile import reconcile_app
from composition.cli.commands.runserver import runserver
from composition.cli.commands.worker import worker


class CLI:
    """CLI: Точка входа для всех команд."""

    def __init__(self) -> None:
        self.cli: Typer = Typer(help="RAG Assistant — CLI")
        self.register_commands()

    def register(
        self,
        name: str,
        func: Callable[..., object],
        help_text: str | None = None,
    ) -> None:
        """Зарегистрировать функцию как CLI-команду."""
        self.cli.command(name=name, help=help_text, add_help_option=True)(func)

    def register_commands(self) -> None:
        """Зарегистрировать все команды."""
        # Сервер
        self.register("runserver", runserver, help_text="Запустить uvicorn-сервер")

        # Arq worker
        self.register("worker", worker, help_text="Запустить Arq worker для фоновых задач")

        # Индексация документов
        self.cli.add_typer(ingest_app, name="ingest")

        # Reconcile Qdrant/Postgres
        self.cli.add_typer(reconcile_app, name="reconcile")

        # Конфигурация
        self.cli.add_typer(config_app, name="config")
        self.cli.add_typer(benchmark_app, name="benchmark")

    def execute_command(self, args: Sequence[str] | None = None) -> None:
        """Выполнить CLI-команду."""
        self.cli(args=args)


cli: CLI = CLI()
