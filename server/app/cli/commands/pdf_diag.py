"""
CLI-команда: диагностика PDF файлов перед индексацией.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer
from pdf_diag import check_pdf

logger = logging.getLogger("cli")

pdf_diag_app = typer.Typer(help="Диагностика PDF файлов перед индексацией")


@pdf_diag_app.command("run")
def pdf_diag_run(
    path: str = typer.Argument(..., help="Путь к PDF файлу или папке с PDF"),
    dump: bool = typer.Option(False, "--dump", help="Показать полный извлечённый текст"),
    chunk_size: int = typer.Option(512, "--chunk-size", help="Размер чанка"),
    chunk_overlap: int = typer.Option(128, "--chunk-overlap", help="Перекрытие чанков"),
) -> None:
    """Проверить PDF на читаемость, типы страниц (текст/скан), качество текста."""
    p = Path(path)

    if p.is_file():
        if p.suffix.lower() != ".pdf":
            typer.echo(f"Файл не является PDF: {p}", err=True)
            sys.exit(1)
        check_pdf(p, dump=dump, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    elif p.is_dir():
        pdfs = list(p.glob("**/*.pdf")) + list(p.glob("**/*.PDF"))
        if not pdfs:
            typer.echo(f"PDF файлов не найдено в {p}", err=True)
            sys.exit(1)

        typer.echo(f"Найдено PDF: {len(pdfs)}")
        results = []
        for pdf in sorted(pdfs):
            r = check_pdf(pdf, dump=dump, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            results.append(r)

        # Общий итог по папке
        typer.echo(f"\n{'═' * 60}")
        typer.echo("СВОДКА ПО ПАПКЕ")
        typer.echo(f"{'═' * 60}")
        total_ok = sum(1 for r in results if r["n_scan"] == 0 and r["n_garbled"] == 0)
        total_scan = sum(1 for r in results if r["n_scan"] > 0)
        total_garb = sum(1 for r in results if r["n_garbled"] > 0)
        total_chars = sum(r["total_chars"] for r in results)
        typer.echo(f"  Читаются нормально: {total_ok}/{len(results)}")
        if total_scan:
            typer.echo(f"  Содержат сканы:     {total_scan}  ← нужен OCR")
        if total_garb:
            typer.echo(f"  Мусорный текст:     {total_garb}  ← нужна конвертация")
        typer.echo(f"  Итого символов:     {total_chars:,}")
    else:
        typer.echo(f"Путь не найден: {p}", err=True)
        sys.exit(1)
