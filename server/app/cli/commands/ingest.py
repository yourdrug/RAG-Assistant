"""
CLI-команда: индексация документов в Qdrant.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer
from ingestion import list_registry, load_registry, run_ingest_file, run_ingestion, save_registry

logger = logging.getLogger("cli")

ingest_app = typer.Typer(help="Индексация документов в Qdrant")


@ingest_app.command("run")
def ingest_run(
    docs_dir: str = typer.Option(
        str(Path("/code/project/data") / "docs_sample"),
        "--docs-dir",
        "-d",
        help="Папка с документами",
    ),
    reset: bool = typer.Option(False, "--reset", help="Сбросить коллекцию и реестр, переиндексировать всё"),
) -> None:
    """Полная индексация папки с документами."""
    try:
        run_ingestion(docs_dir, reset=reset)
    except Exception as exc:
        logger.error("Ошибка при индексации", exc_info=exc)
        sys.exit(1)


@ingest_app.command("file")
def ingest_file(
    file_path: str = typer.Argument(..., help="Путь к файлу внутри контейнера"),
    force: bool = typer.Option(False, "--force", help="Переиндексировать даже если файл уже в реестре"),
) -> None:
    """Добавить один файл в существующую коллекцию."""
    if force:
        registry = load_registry()
        registry.pop(Path(file_path).name, None)
        save_registry(registry)
    try:
        run_ingest_file(file_path)
    except Exception as exc:
        logger.error("Ошибка при индексации файла", exc_info=exc)
        sys.exit(1)


@ingest_app.command("list")
def ingest_list() -> None:
    """Показать список проиндексированных файлов."""
    list_registry()
