"""Evaluate one generated benchmark answer independently of scheduling and report storage."""

from __future__ import annotations

import asyncio
import time

from config import settings
from domain.value_objects.chat_context import ChatContext
from domain.value_objects.benchmark_annotations import validate_annotations
from domain.services.benchmark_evaluation import evaluate_evidence

from infrastructure.benchmark.answer_generators import BenchmarkAnswerGenerator


class BenchmarkCaseEvaluator:
    def __init__(self, generator: BenchmarkAnswerGenerator, judge_model: str) -> None:
        self._generator = generator
        self._judge_model = judge_model

    async def evaluate(self, idx: int, question: dict, run_idx: int, ctx: ChatContext) -> dict:
        from infrastructure.benchmark.judge import judge_answer_async
        from infrastructure.benchmark.metrics import _estimate_cost_usd, compute_context_precision_recall

        annotations = validate_annotations(question.get("annotations"))
        started = time.monotonic()
        generated = await self._generator.generate(question, ctx)
        generator_metrics = await judge_answer_async(
            question=question["question"],
            answer=generated.answer,
            context=generated.context,
            expected_answer=question.get("expected_answer"),
            judge_model=self._judge_model,
        )
        context_metrics = (
            await asyncio.to_thread(
                compute_context_precision_recall,
                question["question"],
                generated.answer,
                context_override=generated.context,
                judge_model=self._judge_model,
            )
            if generated.context
            else {"context_precision": None, "context_recall": None}
        )
        evidence = generated.evidence
        evidence_metrics, diagnostics = evaluate_evidence(
            annotations,
            evidence.retrieved if evidence else None,
            evidence.selected if evidence else None,
            evidence.context if evidence else None,
            generated.answer,
        )
        from infrastructure.benchmark.evidence_judge import judge_evidence

        judge_metrics = await asyncio.to_thread(
            judge_evidence,
            question["question"],
            generated.answer,
            evidence.context if evidence else None,
            annotations,
            self._judge_model,
        )
        evidence_metrics.update(judge_metrics["scores"])
        diagnostics["judge"] = judge_metrics["details"]
        cost = _estimate_cost_usd(
            settings.llm_model, generated.input_tokens or 0, generated.output_tokens or 0
        )
        return {
            "id": question.get("id", str(idx)),
            "question": question["question"],
            "answer": generated.answer,
            "expected_answer": question.get("expected_answer"),
            "source_hint": question.get("source_hint"),
            "retriever_metrics": generated.retriever_metrics,
            "annotations": annotations,
            "evidence_metrics": evidence_metrics,
            "evidence_diagnostics": diagnostics,
            "evidence": {
                "retrieved": evidence.retrieved if evidence else None,
                "retrieval_k": len(evidence.retrieved)
                if evidence and evidence.retrieved is not None
                else None,
                "selected": evidence.selected if evidence else None,
                "context": evidence.context if evidence else None,
                "context_chars": len(evidence.context) if evidence and evidence.context is not None else None,
            },
            "generator_metrics": generator_metrics,
            "context_metrics": context_metrics,
            "latency_sec": round(time.monotonic() - started, 2),
            "input_tokens": generated.input_tokens,
            "output_tokens": generated.output_tokens,
            "cost_usd": round(cost, 6),
            "ttft_sec": generated.ttft_sec,
            "breadth": generated.breadth,
            "domain": generated.domain,
            "run": run_idx,
        }
