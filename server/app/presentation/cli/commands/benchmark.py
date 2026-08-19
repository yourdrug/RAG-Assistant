"""CLI command: RAG quality benchmark with retrieval and LLM-judge scoring."""

from __future__ import annotations

import asyncio
import concurrent.futures
import itertools
import json
import logging
import sys
from pathlib import Path

import typer
from config import settings
from infrastructure.ml.benchmark import (
    load_questions,
    run_benchmark_async,
)
from infrastructure.ml.benchmark_history import (
    compare_runs,
    get_last_baseline,
    load_history,
    print_history,
)
from infrastructure.ml.factories import create_embeddings, create_qdrant_client, load_bm25_index
from infrastructure.ml.hybrid import content_hash, rrf_merge
from infrastructure.services.benchmark_service import BenchmarkService
from langchain.schema import Document as LCDocument

logger = logging.getLogger("cli")

benchmark_app = typer.Typer(help="RAG quality benchmark (retriever + LLM-judge)")


def _parse_comma_separated_ints(values: str) -> list[int]:
    return [int(x.strip()) for x in values.split(",")]


def _parse_comma_separated_floats(values: str) -> list[float]:
    return [float(x.strip()) for x in values.split(",")]


async def _cache_dense_sparse_candidates(
    questions_data: list[dict],
    max_fetch_k: int,
) -> tuple[dict[str, list], dict[str, list], dict[str, LCDocument]]:
    dense_cache: dict[str, list] = {}
    sparse_cache: dict[str, list] = {}
    all_candidates_by_hash: dict[str, LCDocument] = {}

    for q in questions_data:
        qtext = q["question"]
        dense_results = []
        for point in create_qdrant_client().search(
            collection_name=settings.collection_name,
            query_vector=create_embeddings().embed_query(qtext),
            limit=max_fetch_k,
        ):
            payload = point.payload or {}
            page_content = payload.get("page_content", "")
            metadata = payload.get("metadata", {})
            h = metadata.get("content_hash") or content_hash(page_content)
            doc = LCDocument(page_content=page_content, metadata=metadata)
            dense_results.append((h, point.score, doc))
            all_candidates_by_hash[h] = doc
        dense_cache[qtext] = dense_results

        bm25 = load_bm25_index()
        sparse_results = await_if_needed(bm25.search_with_hashes, qtext, max_fetch_k) if bm25 else []
        sparse_cache[qtext] = sparse_results

    logger.info(
        "Phase 1 done: %d dense, %d sparse, %d unique hashes",
        sum(len(v) for v in dense_cache.values()),
        sum(len(v) for v in sparse_cache.values()),
        len(all_candidates_by_hash),
    )
    return dense_cache, sparse_cache, all_candidates_by_hash


def _score_single_question(
    q: dict,
    fetch_k: int,
    top_k: int,
    dw: float,
    sw: float,
    rrf_k: int,
    dense_cache: dict[str, list],
    sparse_cache: dict[str, list],
    all_candidates_by_hash: dict[str, LCDocument],
) -> tuple[int, float]:
    qtext = q["question"]
    source_hint = q.get("source_hint")
    if source_hint is None:
        return 0, 0.0

    dense_trimmed = dense_cache.get(qtext, [])[:fetch_k]
    sparse_trimmed = sparse_cache.get(qtext, [])[:fetch_k]

    merged_hashes = rrf_merge(
        [(h, score) for h, score, _doc in dense_trimmed],
        sparse_trimmed,
        k=rrf_k,
        dense_weight=dw,
        sparse_weight=sw,
    )

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

    return hit, mrr


def _score_retrieval_combos(
    retrieval_combos: list[tuple],
    questions_data: list[dict],
    dense_cache: dict[str, list],
    sparse_cache: dict[str, list],
    all_candidates_by_hash: dict[str, LCDocument],
) -> list[dict]:
    results_summary = []

    for idx, (top_k, fetch_k, dw, sw, rrf_k) in enumerate(retrieval_combos, 1):
        hit_rates_all = []
        mrrs_all = []

        for q in questions_data:
            hit, mrr = _score_single_question(
                q,
                fetch_k,
                top_k,
                dw,
                sw,
                rrf_k,
                dense_cache,
                sparse_cache,
                all_candidates_by_hash,
            )
            hit_rates_all.append(hit)
            mrrs_all.append(mrr)

        avg_hr = sum(hit_rates_all) / len(hit_rates_all) if hit_rates_all else 0
        avg_mrr = sum(mrrs_all) / len(mrrs_all) if mrrs_all else 0

        results_summary.append(
            {
                "top_k": top_k,
                "fetch_k": fetch_k,
                "dense_weight": dw,
                "sparse_weight": sw,
                "rrf_k": rrf_k,
                "avg_hit_rate": round(avg_hr, 3),
                "avg_mrr": round(avg_mrr, 4),
            }
        )

        if (idx % 54) == 0:
            logger.info("  %d/%d done", idx, len(retrieval_combos))

    results_summary.sort(key=lambda x: (x["avg_hit_rate"], x["avg_mrr"]), reverse=True)
    return results_summary


def _apply_config_to_settings(cfg: dict) -> None:
    settings.retriever_top_k = cfg["top_k"]
    settings.retriever_fetch_k = cfg["fetch_k"]
    settings.dense_weight = cfg["dense_weight"]
    settings.sparse_weight = cfg["sparse_weight"]
    settings.rrf_k = cfg["rrf_k"]


def _restore_settings(orig: dict) -> None:
    settings.retriever_top_k = orig["top_k"]
    settings.retriever_fetch_k = orig["fetch_k"]
    settings.dense_weight = orig["dense"]
    settings.sparse_weight = orig["sparse"]
    settings.rrf_k = orig["rrf"]
    settings.rerank_min_score = orig["min_score"]
    settings.rerank_score_gap_ratio = orig["gap_ratio"]


def _collect_candidate_docs(
    qtext: str,
    fetch_k: int,
    rrf_k: int,
    dw: float,
    sw: float,
    dense_cache: dict[str, list],
    sparse_cache: dict[str, list],
    all_candidates_by_hash: dict[str, LCDocument],
) -> list[LCDocument]:
    dense_trimmed = dense_cache.get(qtext, [])[:fetch_k]
    sparse_trimmed = sparse_cache.get(qtext, [])[:fetch_k]

    merged_hashes = rrf_merge(
        [(h, score) for h, score, _doc in dense_trimmed],
        sparse_trimmed,
        k=rrf_k,
        dense_weight=dw,
        sparse_weight=sw,
    )

    seen_h = set()
    candidate_docs = []
    for h in merged_hashes:
        if h not in seen_h:
            seen_h.add(h)
            doc = all_candidates_by_hash.get(h)
            if doc is not None:
                candidate_docs.append(doc)
            if len(candidate_docs) >= fetch_k:
                break
    return candidate_docs


def _apply_rerank_filters(
    ranked: list,
    min_sc: float | None,
    gap_rt: float | None,
) -> list:
    if min_sc is not None:
        ranked = [(d, s) for d, s in ranked if s >= min_sc]
    if gap_rt is not None and ranked:
        top_score = ranked[0][1]
        if top_score > 0:
            ranked = [(d, s) for d, s in ranked if s >= top_score * gap_rt]
    return ranked


def _score_reranked_question(
    q: dict,
    cfg: dict,
    min_sc: float | None,
    gap_rt: float | None,
    dense_cache: dict[str, list],
    sparse_cache: dict[str, list],
    all_candidates_by_hash: dict[str, LCDocument],
    reranker,
) -> tuple[int, float]:
    qtext = q["question"]
    source_hint = q["source_hint"]

    candidate_docs = _collect_candidate_docs(
        qtext,
        cfg["fetch_k"],
        cfg["rrf_k"],
        cfg["dense_weight"],
        cfg["sparse_weight"],
        dense_cache,
        sparse_cache,
        all_candidates_by_hash,
    )

    pairs = []
    for doc in candidate_docs:
        fname = doc.metadata.get("filename", "") or doc.metadata.get("source", "")
        prefix = f"[{fname}] " if fname else ""
        pairs.append((qtext, prefix + doc.page_content))

    if not pairs:
        return 0, 0.0

    scores = reranker.predict(pairs)
    ranked = sorted(
        zip(candidate_docs, scores, strict=False),
        key=lambda x: x[1],
        reverse=True,
    )

    ranked = _apply_rerank_filters(ranked, min_sc, gap_rt)
    top_reranked = ranked[: cfg["top_k"]]

    hit = 0
    mrr = 0.0
    for rank, (doc, _s) in enumerate(top_reranked, 1):
        fname = doc.metadata.get("filename", "") or doc.metadata.get("source", "")
        if source_hint.lower() in fname.lower():
            hit = 1
            if mrr == 0.0:
                mrr = 1.0 / rank
            break

    return hit, mrr


def _find_best_rerank_params(
    eval_questions: list[dict],
    cfg: dict,
    avg_faith: float,
    avg_rel: float,
    rerank_min_list: list[float],
    rerank_gap_list: list[float],
    dense_cache: dict[str, list],
    sparse_cache: dict[str, list],
    all_candidates_by_hash: dict[str, LCDocument],
    reranker,
) -> tuple[float, dict]:
    best_composite = 0.0
    best_params = {"rerank_min_score": None, "rerank_score_gap_ratio": None}

    for min_sc in rerank_min_list:
        for gap_rt in rerank_gap_list:
            rerank_hits = []
            rerank_mrrs = []

            for q in eval_questions:
                hit, mrr = _score_reranked_question(
                    q,
                    cfg,
                    min_sc,
                    gap_rt,
                    dense_cache,
                    sparse_cache,
                    all_candidates_by_hash,
                    reranker,
                )
                rerank_hits.append(hit)
                rerank_mrrs.append(mrr)

            avg_rerank_hr = sum(rerank_hits) / len(rerank_hits) if rerank_hits else 0
            composite = 0.4 * avg_rerank_hr + 0.3 * avg_faith / 10 + 0.3 * avg_rel / 10

            if composite > best_composite:
                best_composite = composite
                best_params = {
                    "rerank_min_score": min_sc,
                    "rerank_score_gap_ratio": gap_rt,
                }

    return best_composite, best_params


def _evaluate_top_configs(
    top_configs: list[dict],
    questions_data: list[dict],
    dense_cache: dict[str, list],
    sparse_cache: dict[str, list],
    all_candidates_by_hash: dict[str, LCDocument],
    rerank_min_list: list[float],
    rerank_gap_list: list[float],
    judge_model: str,
    questions: str,
    out: str,
    service,
    reranker,
) -> None:
    orig = {
        "top_k": settings.retriever_top_k,
        "fetch_k": settings.retriever_fetch_k,
        "dense": settings.dense_weight,
        "sparse": settings.sparse_weight,
        "rrf": settings.rrf_k,
        "min_score": settings.rerank_min_score,
        "gap_ratio": settings.rerank_score_gap_ratio,
    }

    try:
        for config_idx, cfg in enumerate(top_configs, 1):
            logger.info(
                "\n--- LLM evaluation #%d: top_k=%d fetch_k=%d dw=%.1f sw=%.1f rrf_k=%d ---",
                config_idx,
                cfg["top_k"],
                cfg["fetch_k"],
                cfg["dense_weight"],
                cfg["sparse_weight"],
                cfg["rrf_k"],
            )

            _apply_config_to_settings(cfg)

            result = service.run(
                questions_path=questions,
                out_dir=out,
                top_k=cfg["top_k"],
                judge_model=judge_model,
            )

            avg_faith = result.get("avg_faithfulness", 0) or 0
            avg_rel = result.get("avg_relevancy", 0) or 0

            cfg["avg_faithfulness"] = round(avg_faith, 1)
            cfg["avg_relevancy"] = round(avg_rel, 1)

            eval_questions = [q for q in questions_data if q.get("source_hint") is not None]

            best_composite, best_params = _find_best_rerank_params(
                eval_questions,
                cfg,
                avg_faith,
                avg_rel,
                rerank_min_list,
                rerank_gap_list,
                dense_cache,
                sparse_cache,
                all_candidates_by_hash,
                reranker,
            )

            cfg["rerank_min_score"] = best_params["rerank_min_score"]
            cfg["rerank_score_gap_ratio"] = best_params["rerank_score_gap_ratio"]
            cfg["composite_score"] = round(best_composite, 3)

            logger.info(
                "  HR=%.3f  Faith=%.1f  Rel=%.1f  min_sc=%.2f  gap=%.2f  Composite=%.3f",
                cfg["avg_hit_rate"],
                avg_faith,
                avg_rel,
                best_params["rerank_min_score"] or 0,
                best_params["rerank_score_gap_ratio"] or 0,
                best_composite,
            )
    finally:
        _restore_settings(orig)


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
        help="Model for LLM judge (Ollama or OpenRouter)",
    ),
    async_mode: bool = typer.Option(
        False,
        "--async",
        help="Run questions in parallel (requires OLLAMA_NUM_PARALLEL > 1)",
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
            service = BenchmarkService()
            service.run(
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
        help="Model for LLM judge (Ollama or OpenRouter)",
    ),
    top_n_llm: int = typer.Option(
        3,
        "--top-n-llm",
        help="Сколько лучших конфигураций проверять с LLM-судьёй",
    ),
    top_k_values: str = typer.Option(
        "4,6,8,10",
        "--top-k-values",
        help="Через запятую: значения top_k для перебора",
    ),
    fetch_k_values: str = typer.Option(
        "20,30,40",
        "--fetch-k-values",
        help="Через запятую: значения fetch_k (сколько кандидатов до rerank)",
    ),
    dense_weight_values: str = typer.Option(
        "0.5,1.0,1.5",
        "--dense-weight-values",
        help="Через запятую: значения dense_weight",
    ),
    sparse_weight_values: str = typer.Option(
        "0.5,1.0,1.5",
        "--sparse-weight-values",
        help="Через запятую: значения sparse_weight",
    ),
    rrf_k_values: str = typer.Option(
        "30,60,90",
        "--rrf-k-values",
        help="Через запятую: значения rrf_k",
    ),
    rerank_min_score_values: str = typer.Option(
        "0.05,0.10,0.15,0.20",
        "--rerank-min-score-values",
        help="Через запятую: значения rerank_min_score (фильтр по абсолютному порогу)",
    ),
    rerank_gap_ratio_values: str = typer.Option(
        "0.05,0.10,0.20",
        "--rerank-gap-ratio-values",
        help="Через запятую: значения rerank_score_gap_ratio (фильтр по зазору от лучшего)",
    ),
) -> None:
    """Grid search: быстрый retrieval-scoring для всех комбинаций, LLM-судья + rerank для топ-N."""
    from infrastructure.ml.factories import create_reranker

    top_k_list = _parse_comma_separated_ints(top_k_values)
    fetch_k_list = _parse_comma_separated_ints(fetch_k_values)
    dense_weight_list = _parse_comma_separated_floats(dense_weight_values)
    sparse_weight_list = _parse_comma_separated_floats(sparse_weight_values)
    rrf_k_list = _parse_comma_separated_ints(rrf_k_values)
    rerank_min_list = _parse_comma_separated_floats(rerank_min_score_values)
    rerank_gap_list = _parse_comma_separated_floats(rerank_gap_ratio_values)

    # Phase 2: retrieval-only combinations
    retrieval_combos = list(
        itertools.product(top_k_list, fetch_k_list, dense_weight_list, sparse_weight_list, rrf_k_list)
    )

    logger.info("Grid Search — Phase 2: %d retrieval комбинаций", len(retrieval_combos))
    logger.info("  top_k: %s", top_k_list)
    logger.info("  fetch_k: %s", fetch_k_list)
    logger.info("  dense_weight: %s", dense_weight_list)
    logger.info("  sparse_weight: %s", sparse_weight_list)
    logger.info("  rrf_k: %s", rrf_k_list)
    logger.info("  rerank_min_score (Phase 3): %s", rerank_min_list)
    logger.info("  rerank_score_gap_ratio (Phase 3): %s", rerank_gap_list)

    questions_data = load_questions(questions)

    # Phase 1: cache dense+sparse candidates at max fetch_k
    max_fetch_k = max(fetch_k_list)
    logger.info(
        "Phase 1: Кэширую dense + sparse кандидатов (fetch_k=%d) для %d вопросов...",
        max_fetch_k,
        len(questions_data),
    )
    dense_cache, sparse_cache, all_candidates_by_hash = await _cache_dense_sparse_candidates(
        questions_data, max_fetch_k
    )

    # Phase 2: fast retrieval-only scoring
    logger.info("Phase 2: Retrieval-scoring для %d комбинаций...", len(retrieval_combos))
    results_summary = _score_retrieval_combos(
        retrieval_combos, questions_data, dense_cache, sparse_cache, all_candidates_by_hash
    )

    logger.info("Phase 2 done. Top-5 by retrieval:")
    for i, r in enumerate(results_summary[:5], 1):
        logger.info(
            "  #%d  top_k=%d fetch_k=%d dw=%.1f sw=%.1f rrf_k=%d  HR=%.3f MRR=%.4f",
            i,
            r["top_k"],
            r["fetch_k"],
            r["dense_weight"],
            r["sparse_weight"],
            r["rrf_k"],
            r["avg_hit_rate"],
            r["avg_mrr"],
        )

    # Phase 3: full LLM+judge + reranker filtering on top-N configs
    logger.info("Phase 3: LLM-судья + rerank для топ-%d конфигураций...", top_n_llm)
    top_configs = results_summary[:top_n_llm]
    service = BenchmarkService()
    reranker = create_reranker()

    _evaluate_top_configs(
        top_configs,
        questions_data,
        dense_cache,
        sparse_cache,
        all_candidates_by_hash,
        rerank_min_list,
        rerank_gap_list,
        judge_model,
        questions,
        out,
        service,
        reranker,
    )

    # Sort by composite score (with rerank)
    top_configs.sort(key=lambda x: x.get("composite_score", 0), reverse=True)

    # Save grid search results
    grid_path = Path(out) / "grid_search_results.json"
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    grid_path.write_text(
        json.dumps(
            {
                "best_params": {
                    k: v
                    for k, v in (top_configs[0] if top_configs else {}).items()
                    if k
                    in (
                        "top_k",
                        "fetch_k",
                        "dense_weight",
                        "sparse_weight",
                        "rrf_k",
                        "rerank_min_score",
                        "rerank_score_gap_ratio",
                    )
                },
                "best_score": top_configs[0].get("composite_score", 0) if top_configs else 0,
                "all_results": results_summary,
                "llm_evaluated_results": top_configs,
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
        best = top_configs[0]
        logger.info(
            "Лучшие параметры: %s",
            {
                k: v
                for k, v in best.items()
                if k
                in (
                    "top_k",
                    "fetch_k",
                    "dense_weight",
                    "sparse_weight",
                    "rrf_k",
                    "rerank_min_score",
                    "rerank_score_gap_ratio",
                )
            },
        )
        logger.info("Лучший composite score: %.3f", best.get("composite_score", 0))
    logger.info("Результаты сохранены: %s", grid_path)


def await_if_needed(func, *args, **kwargs):
    """Run sync function; if in async context, use asyncio.to_thread."""
    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
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
        help="Model for LLM judge (Ollama or OpenRouter)",
    ),
) -> None:
    """Run benchmark and compare with baseline. Exit code 1 if regression detected."""
    baseline = get_last_baseline(settings.data_dir)

    if baseline:
        logger.info("Baseline found: %s", baseline.get("timestamp", "?"))
    else:
        logger.info("No baseline found — this run will become the baseline.")

    try:
        service = BenchmarkService()
        service.run(
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
