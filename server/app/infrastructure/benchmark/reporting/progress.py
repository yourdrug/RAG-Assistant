"""Benchmark progress and summary logging."""

import logging

from domain.benchmark.evaluation import EVIDENCE_METRICS
from infrastructure.benchmark.evaluation.metrics import compute_summary_metrics

logger = logging.getLogger("default")


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
