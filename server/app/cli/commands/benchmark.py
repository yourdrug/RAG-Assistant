"""
CLI-команда: бенчмарк RAG-системы.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer
from benchmark import run_benchmark
from config import settings

logger = logging.getLogger("cli")

benchmark_app = typer.Typer(help="Оценка качества RAG-системы (retriever + LLM-судья)")


@benchmark_app.command("run")
def benchmark_run(
    questions: str = typer.Option(
        str(Path(settings.data_dir) / "test_questions.json"),
        "--questions",
        "-q",
        help="Путь к JSON-файлу с вопросами",
    ),
    out: str = typer.Option(
        str(Path(settings.data_dir) / "benchmark_results"),
        "--out",
        "-o",
        help="Папка для сохранения результатов",
    ),
    top_k: int = typer.Option(
        settings.retriever_top_k,
        "--top-k",
        "-k",
        help="Количество чанков для retriever",
    ),
    judge_model: str = typer.Option(
        settings.llm_model,
        "--judge-model",
        "-j",
        help="Модель Ollama для роли судьи",
    ),
) -> None:
    """Запустить бенчмарк: retriever-метрики + LLM-судья (faithfulness, relevancy, correctness)."""
    try:
        run_benchmark(
            questions_path=questions,
            out_dir=out,
            top_k=top_k,
            judge_model=judge_model,
        )
    except Exception as exc:
        logger.error("Ошибка при запуске бенчмарка", exc_info=exc)
        sys.exit(1)
