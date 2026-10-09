"""Evaluate one generated benchmark answer independently of scheduling and report storage."""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from infrastructure.benchmark.answer_generators import BenchmarkAnswer
from infrastructure.ml.rag.benchmark_evidence import BenchmarkEvidence
from application.ports.benchmark_checkpoints import BenchmarkCheckpoints
from infrastructure.benchmark.checkpoint import FileBenchmarkCheckpoints
from infrastructure.benchmark.judge_rubric import JUDGE_RUBRIC_VERSION

from config import settings
from domain.value_objects.chat_context import ChatContext
from domain.value_objects.benchmark_annotations import validate_annotations
from domain.services.benchmark_evaluation import evaluate_evidence, evaluate_retrieval

from infrastructure.benchmark.answer_generators import BenchmarkAnswerGenerator
from infrastructure.ml.usage_capture import summarize_usage


class BenchmarkCaseEvaluator:
    def __init__(
        self,
        generator: BenchmarkAnswerGenerator,
        judge_model: str,
        checkpoints: BenchmarkCheckpoints | None = None,
    ) -> None:
        self._generator = generator
        self._judge_model = judge_model
        self._checkpoints = checkpoints

    async def load_stages(self, path: Path | None) -> dict:
        if path is None:
            return {}
        store = self._checkpoints or FileBenchmarkCheckpoints(path.parent)
        return await store.load(str(path) if self._checkpoints else path.name) or {}

    async def save_stages(self, path: Path | None, stages: dict) -> None:
        if path is not None:
            store = self._checkpoints or FileBenchmarkCheckpoints(path.parent)
            await store.save(str(path) if self._checkpoints else path.name, stages)

    async def evaluate(
        self, idx: int, question: dict, run_idx: int, ctx: ChatContext, checkpoint_path: Path | None = None
    ) -> dict:
        from infrastructure.benchmark.token_usage import judge_usage, judge_context

        saved = await self.load_stages(checkpoint_path)
        context_token = judge_context.set(
            {
                "case_id": uuid4().hex,
                "question_id": question.get("id", idx),
                "run": run_idx,
                "checkpoint": str(checkpoint_path) if checkpoint_path else None,
            }
        )
        records = saved.get('judge_usage', judge_usage.get())
        if records is None:
            records = []
        token = judge_usage.set(records)
        try:
            result = await self.evaluate_case(idx, question, run_idx, ctx, checkpoint_path)
            result["judge_input_tokens"] = sum(r.get("input_tokens") or 0 for r in records)
            result["judge_output_tokens"] = sum(r.get("output_tokens") or 0 for r in records)
            result["judge_calls"] = len(records)
            all_records = (result.get('rag_usage') or []) + records
            result['judge_usage'] = records
            result['total_usage'] = summarize_usage(all_records)
            result['total_cost_usd'] = (
                result['total_usage']['cost_usd'] if result.get('rag_usage') is not None else None
            )
            result['cost_usd'] = result['total_cost_usd']
            result["usage_scope"] = (
                "observed RAG model calls plus judge responses, including validation retries"
            )
            return result
        finally:
            judge_usage.reset(token)
            judge_context.reset(context_token)
            if checkpoint_path:
                saved = await self.load_stages(checkpoint_path)
                saved["judge_usage"] = records
                await self.save_stages(checkpoint_path, saved)

    async def evaluate_case(
        self, idx: int, question: dict, run_idx: int, ctx: ChatContext, checkpoint_path: Path | None = None
    ) -> dict:
        from infrastructure.benchmark.metrics import _estimate_cost_usd

        annotations = validate_annotations(question.get("annotations"))
        started = time.monotonic()
        stages = await self.load_stages(checkpoint_path)
        if refresh_judge_rubric(stages):
            await self.save_stages(checkpoint_path, stages)
        saved_answer = stages.get("generated")
        if saved_answer is not None:
            saved_answer = dict(saved_answer)
            evidence = saved_answer.get("evidence")
            saved_answer["evidence"] = BenchmarkEvidence(**evidence) if evidence is not None else None
            generated = BenchmarkAnswer(**saved_answer)
        else:
            generated = await self._generator.generate(question, ctx)
            stages["generated"] = asdict(generated)
            await self.save_stages(checkpoint_path, stages)

        lock = asyncio.Lock()

        async def save_stage(name: str, value: dict) -> None:
            async with lock:
                stages[name] = value
                from infrastructure.benchmark.token_usage import judge_usage

                stages["judge_usage"] = list(judge_usage.get() or [])
                await self.save_stages(checkpoint_path, stages)

        generator_metrics, context_metrics, judge_metrics = await self.judge_generated(
            question, generated, annotations, stages, save_stage
        )
        evidence = generated.evidence
        evidence_metrics, diagnostics = evaluate_evidence(
            annotations,
            evidence.retrieved if evidence else None,
            evidence.selected if evidence else None,
            evidence.context if evidence else None,
            generated.answer,
        )
        retrieval = evaluate_retrieval(question, evidence.retrieved if evidence else None)
        evidence_metrics.update(
            {
                key: retrieval[key]
                for key in (
                    "fragment_recall_at_k",
                    "fragment_mrr",
                    "retrieval_fact_coverage",
                    "evidence_completion_rr",
                )
            }
        )
        retriever_metrics = {**generated.retriever_metrics, **retrieval}
        if evidence is None or evidence.retrieved is None:
            retriever_metrics["source_hit_rate"] = generated.retriever_metrics.get("hit_rate")
            retriever_metrics["source_mrr"] = generated.retriever_metrics.get("mrr")
        # Legacy file metrics remain explicitly named diagnostics, never an
        # alternative result when annotated evidence is unavailable.
        retriever_metrics.setdefault("avg_similarity", 0.0)
        retriever_metrics.setdefault("retrieved_sources", [])
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
            "retriever_metrics": retriever_metrics,
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
                "exclusions": evidence.exclusions if evidence else [],
                "prompt_candidates": evidence.prompt_candidates if evidence else None,
                "prompt_budget": evidence.prompt_budget if evidence else None,
            },
            "generator_metrics": generator_metrics,
            "judge_rubric_version": JUDGE_RUBRIC_VERSION,
            "context_metrics": context_metrics,
            "latency_sec": round(time.monotonic() - started, 2),
            "rag_latency_sec": generated.rag_latency_sec,
            "rag_usage": generated.llm_usage,
            "rag_cost_usd": summarize_usage(generated.llm_usage)['cost_usd']
            if generated.llm_usage is not None
            else None,
            "input_tokens": generated.input_tokens,
            "output_tokens": generated.output_tokens,
            "estimated_generation_cost_usd": round(cost, 6),
            "ttft_sec": generated.ttft_sec,
            "breadth": generated.breadth,
            "domain": generated.domain,
            "run": run_idx,
        }

    async def judge_generated(
        self, question, generated, annotations, stages, save_stage
    ) -> tuple[dict, dict, dict]:
        from infrastructure.benchmark.judge import judge_answer_async
        from infrastructure.benchmark.metrics import compute_context_precision_recall

        if settings.benchmark_judge_grouped_enabled:
            from infrastructure.benchmark.grouped_judge import judge_case_grouped

            saved_scores = restore_grouped_scores(stages)
            grouped = await judge_case_grouped(
                question,
                generated.answer,
                generated.context,
                evidence_context=generated.evidence.context if generated.evidence else None,
                model=self._judge_model,
                saved=saved_scores,
                callback=lambda value: save_stage("grouped_metrics", value),
            )

            def metric_values(keys: tuple[str, ...]) -> dict:
                values = {}
                for key in keys:
                    detail = grouped.get(key, {})
                    values[key] = detail.get("score")
                    values[f"{key}_reason"] = detail.get("reason", "Оценка недоступна")
                    if "error" in detail:
                        values[f"{key}_error"] = detail["error"]
                return values

            generator_metrics = metric_values(("faithfulness", "relevancy", "correctness"))
            context_metrics = metric_values(("context_precision", "context_recall"))
            evidence_keys = ("citation_support_score", "refusal_score", "requirement_preservation_score")
            judge_metrics = {
                "scores": {key: grouped[key].get("score") for key in evidence_keys if key in grouped},
                "details": {
                    key: {k: v for k, v in grouped[key].items() if k != "score"}
                    for key in evidence_keys
                    if key in grouped
                },
            }
        else:
            generator_metrics = await judge_answer_async(
                question=question["question"],
                answer=generated.answer,
                context=generated.context,
                expected_answer=question.get("expected_answer"),
                judge_model=self._judge_model,
                saved_metrics=stages.get("generator_metrics"),
                metric_callback=lambda value: save_stage("generator_metrics", value),
            )
            loop = asyncio.get_running_loop()

            def save_sync(name: str, value: dict) -> None:
                asyncio.run_coroutine_threadsafe(save_stage(name, value), loop).result()

            context_metrics = (
                await asyncio.to_thread(
                    compute_context_precision_recall,
                    question["question"],
                    question.get("expected_answer") or "",
                    context_override=generated.context,
                    judge_model=self._judge_model,
                    saved_metrics=stages.get("context_metrics"),
                    metric_callback=lambda value: save_sync("context_metrics", value),
                )
                if generated.context
                else {"context_precision": None, "context_recall": None}
            )
            from infrastructure.benchmark.evidence_judge import judge_evidence

            judge_metrics = await asyncio.to_thread(
                judge_evidence,
                question["question"],
                generated.answer,
                generated.evidence.context if generated.evidence else None,
                annotations,
                self._judge_model,
                saved_metrics=stages.get("evidence_judge"),
                metric_callback=lambda value: save_sync("evidence_judge", value),
            )
        return generator_metrics, context_metrics, judge_metrics


def refresh_judge_rubric(stages: dict) -> bool:
    """Rejudge old checkpoints while preserving generated answers and evidence."""
    if stages.get("judge_rubric_version") == JUDGE_RUBRIC_VERSION:
        return False
    for key in ("grouped_metrics", "generator_metrics", "context_metrics", "evidence_judge"):
        stages.pop(key, None)
    stages["judge_rubric_version"] = JUDGE_RUBRIC_VERSION
    return True


def restore_grouped_scores(stages: dict) -> dict:
    from infrastructure.benchmark.grouped_judge import INSTRUCTIONS

    saved_scores = dict(stages.get("grouped_metrics", {}))
    # Import successes from older per-metric checkpoints on first resume.
    for group in ("generator_metrics", "context_metrics"):
        values = stages.get(group, {})
        for key in INSTRUCTIONS:
            if values.get(key) is not None:
                saved_scores.setdefault(
                    key, {"score": values[key], "reason": values.get(f"{key}_reason", "")}
                )
    legacy = stages.get("evidence_judge", {})
    for key, value in legacy.get("scores", {}).items():
        if value is not None:
            saved_scores.setdefault(
                key,
                {"score": value, "reason": legacy.get("details", {}).get(key, {}).get("reason", "")},
            )
    return saved_scores
