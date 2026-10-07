"""Benchmark persistence — save results, logging, history."""

import csv
import json
import logging
import re
from datetime import datetime
from pathlib import Path

from config import settings
from domain.services.benchmark_evaluation import EVIDENCE_METRICS

from infrastructure.benchmark.benchmark_history import save_summary_to_history
from infrastructure.benchmark.metrics import compute_summary_metrics

logger = logging.getLogger("default")


def _sanitize_model_name(model: str) -> str:
    """Replace characters invalid in filenames (e.g. ':') with underscores."""
    return re.sub(r'[\\/:*?"<>|]', "_", model)


def save_results(results: list[dict], out_dir: str, model_name: str = "", run_id: str = ""):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = f"_{_sanitize_model_name(model_name)}" if model_name else ""
    run_tag = f"_{run_id}" if run_id else ""

    json_path = out / f"benchmark_{ts}{model_tag}{run_tag}.json"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = out / f"benchmark_{ts}{model_tag}{run_tag}.csv"
    fields = [
        "id",
        "question",
        "faithfulness",
        "relevancy",
        "correctness",
        "hit_rate",
        "mrr",
        "source_hit_rate",
        "source_mrr",
        "avg_sim",
        "context_precision",
        "context_recall",
        "latency_sec",
        *EVIDENCE_METRICS,
        "context_chars",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            gm = result["generator_metrics"]
            rm = result["retriever_metrics"]
            cm = result.get("context_metrics", {})
            writer.writerow(
                {
                    "id": result["id"],
                    "question": result["question"],
                    **{key: gm.get(key) for key in ("faithfulness", "relevancy", "correctness")},
                    "hit_rate": rm.get("hit_rate"),
                    "mrr": rm.get("mrr"),
                    "source_hit_rate": rm.get("source_hit_rate"),
                    "source_mrr": rm.get("source_mrr"),
                    "avg_sim": rm.get("avg_similarity"),
                    "context_precision": cm.get("context_precision"),
                    "context_recall": cm.get("context_recall"),
                    "latency_sec": round(result["latency_sec"], 2),
                    **{key: result.get("evidence_metrics", {}).get(key) for key in EVIDENCE_METRICS},
                    "context_chars": result.get("evidence", {}).get("context_chars"),
                }
            )

    logger.info("Результаты сохранены:")
    logger.info("  JSON: %s", json_path)
    logger.info("  CSV:  %s", csv_path)

    summary = compute_summary_metrics(results)

    config = {
        "top_k": settings.retriever_top_k,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "tei_embed_url": settings.tei_embed_url,
        "llm_model": model_name or settings.llm_model,
        "hybrid_enabled": settings.hybrid_enabled,
        "dense_weight": settings.dense_weight,
        "sparse_weight": settings.sparse_weight,
        "rrf_k": settings.rrf_k,
    }

    save_summary_to_history(summary, config, settings.data_dir)


def log_question_result(idx: int, total: int, q: dict, result: dict):
    logger.info("[%d/%d] %s", idx, total, q["question"])

    rm = result["retriever_metrics"]
    sim_str = f"avg_sim={rm['avg_similarity']:.3f}"
    if rm["hit_rate"] is not None:
        hr_str = "hit" if rm["hit_rate"] else "miss"
        mrr_str = f"mrr={rm['mrr']:.2f}" if rm["mrr"] is not None else "mrr=n/a"
        logger.info("  Retriever: %s  %s  %s", hr_str, mrr_str, sim_str)
    else:
        logger.info("  Retriever: %s  (оценка недоступна: нет разметки или evidence)", sim_str)

    src_list = ", ".join(result["retriever_metrics"]["retrieved_sources"][:3])
    logger.info("  Источники: %s", src_list)

    gm = result["generator_metrics"]
    logger.info("  Faithfulness: %s/10  — %s", gm["faithfulness"], gm["faithfulness_reason"])
    logger.info("  Relevancy:    %s/10  — %s", gm["relevancy"], gm["relevancy_reason"])
    if gm["correctness"] is not None:
        logger.info("  Correctness:  %s/10  — %s", gm["correctness"], gm["correctness_reason"])

    cm = result.get("context_metrics", {})
    cp = cm.get("context_precision")
    cr = cm.get("context_recall")
    if cp is not None:
        logger.info("  Context Precision: %s/10  — %s", cp, cm.get("context_precision_reason", ""))
    if cr is not None:
        logger.info("  Context Recall:    %s/10  — %s", cr, cm.get("context_recall_reason", ""))

    if result.get("evidence_metrics"):
        logger.info("  Evidence: %s", result["evidence_metrics"])

    answer_preview = result["answer"][:200].replace("\n", " ")
    if len(result["answer"]) > 200:
        answer_preview += "..."
    logger.info("  Ответ: %s", answer_preview)
    logger.info("  Время: %.1fs", result["latency_sec"])


def log_summary(results: list[dict], total_time: float):
    n = len(results)
    logger.info("=" * 60)
    logger.info("ИТОГОВЫЙ ОТЧЁТ  (%d вопросов, %.1fs)", n, total_time)
    logger.info("=" * 60)

    m = compute_summary_metrics(results)

    logger.info("Retriever:")
    if m["hit_rate"] is not None:
        logger.info(
            "  Hit Rate:        %s/10  (%.0f%% размеченных вопросов прошли проверку retrieval)",
            m["hit_rate"] * 10,
            m["hit_rate"] * 100,
        )
        if m["avg_mrr"] is not None:
            logger.info(
                "  MRR:             %.3f  (первый подходящий фрагмент или legacy-источник)", m["avg_mrr"]
            )
    logger.info("  Avg Similarity:  %.3f", m["avg_similarity"])

    logger.info("Context Quality:")
    if m["avg_context_precision"] is not None:
        logger.info(
            "  Context Precision: %s/10  (доля релевантных документов среди ретривированных)",
            m["avg_context_precision"],
        )
    if m["avg_context_recall"] is not None:
        logger.info(
            "  Context Recall:    %s/10  (доля нужной информации в контексте)", m["avg_context_recall"]
        )

    for metric in EVIDENCE_METRICS:
        if m.get(f"avg_{metric}") is not None:
            logger.info(
                "  %s: %.4f (evaluated=%d)", metric, m[f"avg_{metric}"], m[f"{metric}_evaluated_count"]
            )

    logger.info("Generator:")
    logger.info("  Faithfulness:    %s/10  (достоверность — нет ли выдуманных фактов)", m["avg_faithfulness"])
    logger.info("  Relevancy:       %s/10  (ответ по существу вопроса)", m["avg_relevancy"])
    if m["avg_correctness"] is not None:
        logger.info("  Correctness:     %s/10  (совпадение с эталоном)", m["avg_correctness"])

    bad = [
        r
        for r in results
        if any(
            r["generator_metrics"].get(key) is not None and r["generator_metrics"][key] < 5
            for key in ("faithfulness", "relevancy")
        )
    ]
    if bad:
        logger.warning("Проблемные вопросы (%d):", len(bad))
        for r in bad:
            gm = r["generator_metrics"]
            logger.warning("  [%s] %s", r["id"], r["question"][:60])
            logger.warning("        faith=%s  rel=%s", gm["faithfulness"], gm["relevancy"])
            logger.warning("        %s", gm["faithfulness_reason"])

    logger.info("Время: %.1fs  (%.1fs на вопрос)", total_time, total_time / n)
    logger.info("=" * 60)
