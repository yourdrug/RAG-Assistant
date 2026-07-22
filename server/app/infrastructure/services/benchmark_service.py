"""Benchmark Service — runs benchmark and returns structured results."""

from __future__ import annotations

import logging
import time

from config import settings

from infrastructure.ml.benchmark import (
    build_llm,
    build_retriever,
    compute_retriever_metrics,
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
        total_start = time.time()

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

        total_time = time.time() - total_start

        save_results(results, out_dir, model_name=settings.llm_model)

        n = len(results)
        faiths = [r["generator_metrics"]["faithfulness"] for r in results]
        rels = [r["generator_metrics"]["relevancy"] for r in results]
        corrs = [
            r["generator_metrics"]["correctness"]
            for r in results
            if r["generator_metrics"]["correctness"] is not None
        ]
        hit_rates = [
            r["retriever_metrics"]["hit_rate"]
            for r in results
            if r["retriever_metrics"]["hit_rate"] is not None
        ]
        mrrs = [r["retriever_metrics"]["mrr"] for r in results if r["retriever_metrics"]["mrr"] is not None]
        sims = [r["retriever_metrics"]["avg_similarity"] for r in results]

        summary = {
            "total_questions": n,
            "total_time_sec": round(total_time, 1),
            "avg_faithfulness": round(sum(faiths) / len(faiths), 1) if faiths else None,
            "avg_relevancy": round(sum(rels) / len(rels), 1) if rels else None,
            "avg_correctness": round(sum(corrs) / len(corrs), 1) if corrs else None,
            "hit_rate": round(sum(hit_rates) / len(hit_rates), 3) if hit_rates else None,
            "avg_mrr": round(sum(mrrs) / len(mrrs), 3) if mrrs else None,
            "avg_similarity": round(sum(sims) / len(sims), 3) if sims else 0,
            "results": [
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
            ],
        }

        return summary
