"""
CLI-команда: бенчмарк RAG-системы.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import sys
from pathlib import Path

import typer
from config import settings
from infrastructure.clients import get_bm25_index, get_embeddings, get_qdrant_client
from infrastructure.ml.benchmark import (
    load_questions,
    run_benchmark,
    run_benchmark_async,
)
from infrastructure.ml.benchmark_history import compare_runs, get_last_baseline, print_history
from infrastructure.ml.hybrid import content_hash, rrf_merge
from langchain.schema import Document as LCDocument

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
    async_mode: bool = typer.Option(
        False,
        "--async",
        help="Запускать вопросы параллельно (requires OLLAMA_NUM_PARALLEL > 1)",
    ),
    max_concurrent: int = typer.Option(
        4,
        "--max-concurrent",
        help="Макс. параллельных вопросов (при --async)",
    ),
    seed: int | None = typer.Option(
        None,
        "--seed",
        "-s",
        help="Seed для воспроизводимости (передаётся в ChatOllama)",
    ),
    n_runs: int = typer.Option(
        1,
        "--n-runs",
        help="Количество запусков для усреднения (при n_runs>1 результаты усредняются)",
    ),
) -> None:
    """Запустить бенчмарк: retriever-метрики + LLM-судья (faithfulness, relevancy, correctness)."""
    try:
        if async_mode:
            asyncio.run(
                run_benchmark_async(
                    questions_path=questions,
                    out_dir=out,
                    top_k=top_k,
                    judge_model=judge_model,
                    max_concurrent=max_concurrent,
                    seed=seed,
                    n_runs=n_runs,
                )
            )
        else:
            run_benchmark(
                questions_path=questions,
                out_dir=out,
                top_k=top_k,
                judge_model=judge_model,
                seed=seed,
                n_runs=n_runs,
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
    top_n_llm: int = typer.Option(
        3,
        "--top-n-llm",
        help="Сколько лучших конфигураций проверять с LLM-судьёй",
    ),
) -> None:
    """Grid search: быстрый retrieval-scoring для всех комбинаций, LLM-судья только для топ-N."""
    # Parameter grid
    top_k_values = [4, 6, 8, 10]
    dense_weight_values = [0.5, 1.0, 1.5]
    sparse_weight_values = [0.5, 1.0, 1.5]
    rrf_k_values = [30, 60, 90]

    combinations = list(
        itertools.product(top_k_values, dense_weight_values, sparse_weight_values, rrf_k_values)
    )

    logger.info("Grid Search: %d комбинаций (fast retrieval phase)", len(combinations))
    logger.info("  top_k: %s", top_k_values)
    logger.info("  dense_weight: %s", dense_weight_values)
    logger.info("  sparse_weight: %s", sparse_weight_values)
    logger.info("  rrf_k: %s", rrf_k_values)

    questions_data = load_questions(questions)

    # --- Phase 1: cache dense+sparse candidates for all questions (one Qdrant pass each) ---
    logger.info("Phase 1: Кэширую.dense + sparse кандидатов для %d вопросов...", len(questions_data))
    fetch_k = max(settings.retriever_fetch_k, settings.retriever_fetch_k_broad)
    dense_cache: dict[str, list] = {}  # question -> [(hash, score, doc)]
    sparse_cache: dict[str, list] = {}  # question -> [(hash, score)]
    all_candidates_by_hash: dict[str, LCDocument] = {}  # hash -> doc (universal)

    for q in questions_data:
        qtext = q["question"]
        # Dense search
        dense_results = []
        for point in get_qdrant_client().search(
            collection_name=settings.collection_name,
            query_vector=get_embeddings().embed_query(qtext),
            limit=fetch_k,
        ):
            payload = point.payload or {}
            page_content = payload.get("page_content", "")
            metadata = payload.get("metadata", {})
            h = metadata.get("content_hash") or content_hash(page_content)
            doc = LCDocument(page_content=page_content, metadata=metadata)
            dense_results.append((h, point.score, doc))
            all_candidates_by_hash[h] = doc
        dense_cache[qtext] = dense_results

        # Sparse search
        bm25 = get_bm25_index()
        sparse_results = await_if_needed(bm25.search_with_hashes, qtext, fetch_k) if bm25 else []
        sparse_cache[qtext] = sparse_results

    logger.info(
        "Phase 1 done: %d dense, %d sparse, %d unique hashes",
        sum(len(v) for v in dense_cache.values()),
        sum(len(v) for v in sparse_cache.values()),
        len(all_candidates_by_hash),
    )

    # --- Phase 2: fast retrieval-only scoring for all combinations ---
    logger.info("Phase 2: Retrieval-scoring для %d комбинаций...", len(combinations))
    results_summary = []

    for idx, (top_k, dw, sw, rrf_k) in enumerate(combinations, 1):
        hit_rates_all = []
        mrrs_all = []

        for q in questions_data:
            qtext = q["question"]
            source_hint = q.get("source_hint")
            if source_hint is None:
                continue

            merged_hashes = rrf_merge(
                [(h, score) for h, score, _doc in dense_cache.get(qtext, [])],
                sparse_cache.get(qtext, []),
                k=rrf_k,
                dense_weight=dw,
                sparse_weight=sw,
            )

            # Deduplicate + top_k
            seen = set()
            top_hashes = []
            for h in merged_hashes:
                if h not in seen:
                    seen.add(h)
                    top_hashes.append(h)
                    if len(top_hashes) >= top_k:
                        break

            hit = 0
            mrr = 0.0
            for rank, h in enumerate(top_hashes, 1):
                doc = all_candidates_by_hash.get(h)
                if doc is None:
                    continue
                filename = doc.metadata.get("filename", "") or doc.metadata.get("source", "")
                if source_hint.lower() in filename.lower():
                    hit = 1
                    if mrr == 0.0:
                        mrr = 1.0 / rank
                    break

            hit_rates_all.append(hit)
            mrrs_all.append(mrr)

        avg_hr = sum(hit_rates_all) / len(hit_rates_all) if hit_rates_all else 0
        avg_mrr = sum(mrrs_all) / len(mrrs_all) if mrrs_all else 0

        results_summary.append(
            {
                "top_k": top_k,
                "dense_weight": dw,
                "sparse_weight": sw,
                "rrf_k": rrf_k,
                "avg_hit_rate": round(avg_hr, 3),
                "avg_mrr": round(avg_mrr, 4),
            }
        )

        if (idx % 27) == 0:
            logger.info("  %d/%d done", idx, len(combinations))

    # Sort by hit_rate (primary) then MRR (secondary)
    results_summary.sort(key=lambda x: (x["avg_hit_rate"], x["avg_mrr"]), reverse=True)

    logger.info("Phase 2 done. Top-5 by retrieval:")
    for i, r in enumerate(results_summary[:5], 1):
        logger.info(
            "  #%d  top_k=%d dw=%.1f sw=%.1f rrf_k=%d  HR=%.3f MRR=%.4f",
            i,
            r["top_k"],
            r["dense_weight"],
            r["sparse_weight"],
            r["rrf_k"],
            r["avg_hit_rate"],
            r["avg_mrr"],
        )

    # --- Phase 3: full LLM+judge only on top-N configs ---
    logger.info("Phase 3: LLM-судья для топ-%d конфигураций...", top_n_llm)
    top_configs = results_summary[:top_n_llm]

    for config_idx, cfg in enumerate(top_configs, 1):
        logger.info(
            "\n--- LLM evaluation #%d: top_k=%d dw=%.1f sw=%.1f rrf_k=%d ---",
            config_idx,
            cfg["top_k"],
            cfg["dense_weight"],
            cfg["sparse_weight"],
            cfg["rrf_k"],
        )

        original_top_k = settings.retriever_top_k
        original_dense = settings.dense_weight
        original_sparse = settings.sparse_weight
        original_rrf = settings.rrf_k

        try:
            settings.retriever_top_k = cfg["top_k"]
            settings.dense_weight = cfg["dense_weight"]
            settings.sparse_weight = cfg["sparse_weight"]
            settings.rrf_k = cfg["rrf_k"]

            run_benchmark(
                questions_path=questions,
                out_dir=out,
                top_k=cfg["top_k"],
                judge_model=judge_model,
            )

            # Read back the metrics
            result_files = sorted(Path(out).glob("benchmark_*.json"))
            if result_files:
                latest = json.loads(result_files[-1].read_text(encoding="utf-8"))
                faiths = [r["generator_metrics"]["faithfulness"] for r in latest]
                rels = [r["generator_metrics"]["relevancy"] for r in latest]
                avg_faith = sum(faiths) / len(faiths) if faiths else 0
                avg_rel = sum(rels) / len(rels) if rels else 0
                composite = 0.4 * cfg["avg_hit_rate"] + 0.3 * avg_faith / 10 + 0.3 * avg_rel / 10

                cfg["avg_faithfulness"] = round(avg_faith, 1)
                cfg["avg_relevancy"] = round(avg_rel, 1)
                cfg["composite_score"] = round(composite, 3)

                logger.info(
                    "  HR=%.3f  Faith=%.1f  Rel=%.1f  Composite=%.3f",
                    cfg["avg_hit_rate"],
                    avg_faith,
                    avg_rel,
                    composite,
                )
        finally:
            settings.retriever_top_k = original_top_k
            settings.dense_weight = original_dense
            settings.sparse_weight = original_sparse
            settings.rrf_k = original_rrf

    # Save grid search results
    grid_path = Path(out) / "grid_search_results.json"
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    grid_path.write_text(
        json.dumps(
            {
                "best_params": top_configs[0] if top_configs else None,
                "best_score": top_configs[0].get("composite_score", 0) if top_configs else 0,
                "all_results": results_summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info("\n" + "=" * 60)
    logger.info("GRID SEARCH ЗАВЕРШЁН")
    logger.info("=" * 60)
    if top_configs:
        logger.info(
            "Лучшие параметры: %s",
            {
                k: v
                for k, v in top_configs[0].items()
                if k in ("top_k", "dense_weight", "sparse_weight", "rrf_k")
            },
        )
        logger.info("Лучший composite score: %.3f", top_configs[0].get("composite_score", 0))
    logger.info("Результаты сохранены: %s", grid_path)


def await_if_needed(func, *args, **kwargs):
    """Run sync function; if in async context, use asyncio.to_thread."""
    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(func, *args, **kwargs).result()
    except RuntimeError:
        pass
    return func(*args, **kwargs)


@benchmark_app.command("regression")
def benchmark_regression(
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
    """Run benchmark and compare with baseline. Exit code 1 if regression detected."""
    baseline = get_last_baseline(settings.data_dir)

    if baseline:
        logger.info("Baseline found: %s", baseline.get("timestamp", "?"))
    else:
        logger.info("No baseline found — this run will become the baseline.")

    try:
        run_benchmark(
            questions_path=questions,
            out_dir=out,
            top_k=top_k,
            judge_model=judge_model,
        )
    except Exception as exc:
        logger.error("Benchmark failed", exc_info=exc)
        sys.exit(1)

    if baseline is None:
        logger.info("First run saved as baseline. No regression check possible.")
        return

    # Reload history to get the run we just saved
    from infrastructure.ml.benchmark_history import load_history

    history = load_history(settings.data_dir)
    if len(history) < 2:
        logger.info("Not enough history for comparison.")
        return

    current = history[-1]
    comp = compare_runs(current, baseline)

    logger.info("")
    logger.info("=" * 60)
    logger.info("REGRESSION CHECK  (current vs baseline %s)", baseline.get("timestamp", "?"))
    logger.info("=" * 60)
    logger.info(
        "%-20s  %8s  %8s  %8s  %10s  %s",
        "Metric",
        "Baseline",
        "Current",
        "Delta",
        "Threshold",
        "Status",
    )
    logger.info("-" * 60)

    for r in comp["results"]:
        if r["delta"] is not None:
            status = "FAIL" if r["failed"] else "ok"
            logger.info(
                "%-20s  %8.4f  %8.4f  %+.4f  %+.4f  %s",
                r["metric"],
                r["baseline"],
                r["current"],
                r["delta"],
                r["threshold"],
                status,
            )
        else:
            logger.info("%-20s  %8s  %8s  %8s  %10s  %s", r["metric"], "-", "-", "-", "-", r.get("note", ""))

    logger.info("=" * 60)

    if comp["passed"]:
        logger.info("PASSED — no regression detected")
    else:
        logger.error("FAILED — regression detected (thresholds exceeded)")
        sys.exit(1)


@benchmark_app.command("history")
def benchmark_history(
    n: int = typer.Option(10, "--last", "-n", help="Number of recent runs to show"),
) -> None:
    """Show benchmark history — last N runs with trend comparison."""
    print_history(settings.data_dir, n=n)
