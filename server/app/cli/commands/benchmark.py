"""
CLI-команда: бенчмарк RAG-системы.
"""

from __future__ import annotations

import itertools
import logging
import sys
from pathlib import Path

import typer
from config import settings
from infrastructure.ml.benchmark import run_benchmark

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


@benchmark_app.command("grid-search")
def benchmark_grid_search(
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
    judge_model: str = typer.Option(
        settings.llm_model,
        "--judge-model",
        "-j",
        help="Модель Ollama для роли судьи",
    ),
) -> None:
    """Grid search по параметрам retriever: top_k, fetch_k, dense_weight, sparse_weight, rrf_k."""
    # Parameter grid
    top_k_values = [4, 6, 8, 10]
    dense_weight_values = [0.5, 1.0, 1.5]
    sparse_weight_values = [0.5, 1.0, 1.5]
    rrf_k_values = [30, 60, 90]

    combinations = list(itertools.product(
        top_k_values, dense_weight_values, sparse_weight_values, rrf_k_values
    ))

    logger.info("Grid Search: %d комбинаций параметров", len(combinations))
    logger.info("  top_k: %s", top_k_values)
    logger.info("  dense_weight: %s", dense_weight_values)
    logger.info("  sparse_weight: %s", sparse_weight_values)
    logger.info("  rrf_k: %s", rrf_k_values)

    best_score = -1.0
    best_params = None
    best_result = None
    results_summary = []

    for idx, (top_k, dw, sw, rrf_k) in enumerate(combinations, 1):
        logger.info("\n[%d/%d] top_k=%d, dense=%.1f, sparse=%.1f, rrf_k=%d",
                    idx, len(combinations), top_k, dw, sw, rrf_k)

        # Temporarily override settings
        original_top_k = settings.retriever_top_k
        original_dense = settings.dense_weight
        original_sparse = settings.sparse_weight
        original_rrf = settings.rrf_k

        try:
            settings.retriever_top_k = top_k
            settings.dense_weight = dw
            settings.sparse_weight = sw
            settings.rrf_k = rrf_k

            # Run benchmark with these settings
            run_benchmark(
                questions_path=questions,
                out_dir=out,
                top_k=top_k,
                judge_model=judge_model,
            )

            # Read the latest result to get metrics
            result_files = sorted(Path(out).glob("benchmark_*.json"))
            if result_files:
                import json
                latest = json.loads(result_files[-1].read_text(encoding="utf-8"))
                # Calculate average metrics
                hit_rates = [r["retriever_metrics"]["hit_rate"] for r in latest
                             if r["retriever_metrics"]["hit_rate"] is not None]
                faiths = [r["generator_metrics"]["faithfulness"] for r in latest]
                rels = [r["generator_metrics"]["relevancy"] for r in latest]

                avg_hr = sum(hit_rates) / len(hit_rates) if hit_rates else 0
                avg_faith = sum(faiths) / len(faiths) if faiths else 0
                avg_rel = sum(rels) / len(rels) if rels else 0

                # Composite score: weighted combination
                composite = 0.4 * avg_hr + 0.3 * avg_faith/10 + 0.3 * avg_rel/10

                results_summary.append({
                    "top_k": top_k,
                    "dense_weight": dw,
                    "sparse_weight": sw,
                    "rrf_k": rrf_k,
                    "avg_hit_rate": round(avg_hr, 3),
                    "avg_faithfulness": round(avg_faith, 1),
                    "avg_relevancy": round(avg_rel, 1),
                    "composite_score": round(composite, 3),
                })

                logger.info("  Hit Rate: %.1f%%, Faith: %.1f, Rel: %.1f, Score: %.3f",
                            avg_hr * 100, avg_faith, avg_rel, composite)

                if composite > best_score:
                    best_score = composite
                    best_params = {
                        "top_k": top_k,
                        "dense_weight": dw,
                        "sparse_weight": sw,
                        "rrf_k": rrf_k,
                    }
                    best_result = results_summary[-1]

        finally:
            # Restore original settings
            settings.retriever_top_k = original_top_k
            settings.dense_weight = original_dense
            settings.sparse_weight = original_sparse
            settings.rrf_k = original_rrf

    # Save grid search results
    import json
    grid_path = Path(out) / "grid_search_results.json"
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    grid_path.write_text(json.dumps({
        "best_params": best_params,
        "best_score": best_score,
        "all_results": sorted(results_summary, key=lambda x: x["composite_score"], reverse=True),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("\n" + "=" * 60)
    logger.info("GRID SEARCH ЗАВЕРШЁН")
    logger.info("=" * 60)
    logger.info("Лучшие параметры: %s", best_params)
    logger.info("Лучший composite score: %.3f", best_score)
    logger.info("Результаты сохранены: %s", grid_path)
