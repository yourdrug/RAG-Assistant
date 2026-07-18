"""
CLI-команда: индексация документов в Qdrant.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer
from infrastructure.vector_store import (
    list_registry,
    load_registry,
    run_ingest_file,
    run_ingestion,
    save_registry,
)

logger = logging.getLogger("cli")

ingest_app = typer.Typer(help="Индексация документов в Qdrant")


@ingest_app.command("run")
def ingest_run(
    docs_dir: str = typer.Option(
        str(Path("/code/project/data") / "docs_sample"),
        "--docs-dir",
        "-d",
        help="Папка с документами (local режим)",
    ),
    reset: bool = typer.Option(False, "--reset", help="Сбросить коллекцию и реестр, переиндексировать всё"),
    s3: bool = typer.Option(False, "--s3", help="Индексировать из S3 (вместо локальной папки)"),
    prefix: str = typer.Option("docs/", "--prefix", "-p", help="S3 prefix (по умолчанию docs/)"),
) -> None:
    """Полная индексация папки с документами."""
    try:
        import os

        if s3:
            os.environ["FILE_BACKEND"] = "s3"
            from config import settings

            settings.file_backend = "s3"
        run_ingestion(docs_dir, reset=reset, prefix=prefix)
    except Exception as exc:
        logger.error("Ошибка при индексации", exc_info=exc)
        sys.exit(1)


@ingest_app.command("file")
def ingest_file(
    file_path: str = typer.Argument(..., help="Путь к файлу (локальный) или S3 key (docs/report.pdf)"),
    force: bool = typer.Option(False, "--force", help="Переиндексировать даже если файл уже в реестре"),
    s3: bool = typer.Option(False, "--s3", help="Трактовать путь как S3 key"),
) -> None:
    """Добавить один файл в существующую коллекцию."""
    if force:
        registry = load_registry()
        registry.pop(Path(file_path).name, None)
        save_registry(registry)
    if s3:
        import os

        os.environ["FILE_BACKEND"] = "s3"
        from config import settings

        settings.file_backend = "s3"
    try:
        run_ingest_file(file_path)
    except Exception as exc:
        logger.error("Ошибка при индексации файла", exc_info=exc)
        sys.exit(1)


@ingest_app.command("upload")
def ingest_upload(
    file_path: str = typer.Argument(..., help="Локальный путь к файлу для загрузки в S3"),
    key: str = typer.Option(None, "--key", "-k", help="S3 key (по умолчанию docs/<filename>)"),
) -> None:
    """Загрузить локальный файл в S3-хранилище."""
    import os

    os.environ["FILE_BACKEND"] = "s3"
    from config import settings

    settings.file_backend = "s3"

    from infrastructure.storage import get_storage

    path = Path(file_path)
    if not path.exists():
        logger.error("Файл не найден: %s", file_path)
        sys.exit(1)

    storage = get_storage()
    s3_key = key or f"docs/{path.name}"
    data = path.read_bytes()
    storage.upload_file(s3_key, data)
    logger.info("Загружено: s3://%s/%s (%d bytes)", settings.s3_bucket, s3_key, len(data))


@ingest_app.command("list")
def ingest_list() -> None:
    """Показать список проиндексированных файлов."""
    list_registry()
