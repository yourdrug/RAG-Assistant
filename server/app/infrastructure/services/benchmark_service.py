"""Benchmark Service — runs benchmark and returns structured results."""

from __future__ import annotations

import logging
import time

from config import settings

from infrastructure.ml.benchmark import (
    build_llm,
    build_retriever,
    compute_retriever_metrics,
    compute_summary_metrics,
    get_rag_answer,
    judge_answer,
    load_questions,
    retrieve_with_scores,
    save_results,
)

log = logging.getLogger("default")


class BenchmarkService:
    def run(
        self,
        questions_path: str,
        out_dir: str,
        top_k: int,
        judge_model: str,
    ) -> dict:
        log.info("RAG Benchmark (API)")
        log.info("  questions : %s", questions_path)
        log.info("  top_k     : %d", top_k)
        log.info("  rag model : %s", settings.llm_model)
        log.info("  judge     : %s", judge_model)

        questions = load_questions(questions_path)
        retriever, vs = build_retriever(top_k)

        rag_llm = build_llm(settings.llm_model, settings.ollama_base_url)
        judge_llm = build_llm(judge_model, settings.ollama_base_url)

        log.info("Прогрев моделей ...")
        rag_llm.invoke("Привет")
        if judge_model != settings.llm_model:
            judge_llm.invoke("Привет")

        log.info("Запускаю тесты...")
        results = []

        for idx, q in enumerate(questions, 1):
            t_start = time.time()

            docs_with_scores = retrieve_with_scores(vs, q["question"], top_k)
            answer = get_rag_answer(rag_llm, docs_with_scores, q["question"])
            retriever_metrics = compute_retriever_metrics(docs_with_scores, q.get("source_hint"))

            context_for_judge = "\n\n---\n\n".join(d.page_content for d, _ in docs_with_scores)
            generator_metrics = judge_answer(
                judge_llm,
                question=q["question"],
                answer=answer,
                context=context_for_judge,
                expected_answer=q.get("expected_answer"),
            )

            latency = time.time() - t_start

            result = {
                "id": q.get("id", str(idx)),
                "question": q["question"],
                "answer": answer,
                "expected_answer": q.get("expected_answer"),
                "source_hint": q.get("source_hint"),
                "retriever_metrics": retriever_metrics,
                "generator_metrics": generator_metrics,
                "latency_sec": round(latency, 2),
            }
            results.append(result)

        save_results(results, out_dir, model_name=settings.llm_model)

        summary = compute_summary_metrics(results)
        summary["results"] = [
            {
                "id": r["id"],
                "question": r["question"],
                "answer": r["answer"],
                "expected_answer": r["expected_answer"],
                "faithfulness": r["generator_metrics"]["faithfulness"],
                "relevancy": r["generator_metrics"]["relevancy"],
                "correctness": r["generator_metrics"]["correctness"],
                "hit_rate": r["retriever_metrics"]["hit_rate"],
                "mrr": r["retriever_metrics"]["mrr"],
                "avg_similarity": r["retriever_metrics"]["avg_similarity"],
                "latency_sec": r["latency_sec"],
            }
            for r in results
        ]

        return summary
