"""Evaluate one generated benchmark answer independently of scheduling and report storage."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
import time
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from config import settings
from domain.benchmark.evaluation import evaluate_evidence, evaluate_retrieval, evaluate_stage_evidence
from domain.benchmark.question import (
    ABSTENTION_TYPES,
    answer_type,
    evaluation_annotations as normalize_annotations,
    validate_annotations,
)
from domain.benchmark.stage_reporting import diagnostic_card
from infrastructure.benchmark.checkpoint import load_stages, refresh_judge_rubric, save_stages
from infrastructure.benchmark.evaluation import metrics
from infrastructure.benchmark.evaluation.annotation_validation import (
    question_context,
    report_annotations,
)
from infrastructure.benchmark.evaluation.answer_generators import BenchmarkAnswer
from infrastructure.benchmark.evaluation.case_judging import judge_generated
from infrastructure.benchmark.judging.judge_rubric import JUDGE_RUBRIC_VERSION
from infrastructure.benchmark.judging.token_usage import judge_context, judge_usage
from infrastructure.ml.rag.benchmark_evidence import BenchmarkEvidence
from infrastructure.ml.usage_capture import summarize_usage


if TYPE_CHECKING:
    from pathlib import Path

    from application.benchmark import BenchmarkCheckpoints
    from domain.rag.context import RagContext
    from infrastructure.benchmark.evaluation.annotation_validation import (
        AnnotationValidator,
    )
    from infrastructure.benchmark.evaluation.answer_generators import BenchmarkAnswerGenerator
    from infrastructure.benchmark.judging.callbacks import SaveStage
    from infrastructure.budget.runtime import BudgetRuntime


@dataclass(frozen=True)
class CaseRun:
    idx: int
    run_idx: int
    started: float


@dataclass(frozen=True)
class CaseJudgement:
    annotations: dict[str, Any] | None
    generator_metrics: dict[str, Any]
    context_metrics: dict[str, Any]
    judge_metrics: dict[str, Any]


class BenchmarkCaseEvaluator:
    def __init__(
        self,
        generator: BenchmarkAnswerGenerator,
        judge_model: str,
        checkpoints: BenchmarkCheckpoints | None = None,
        *,
        budget_runtime: BudgetRuntime | None = None,
        annotation_validator: AnnotationValidator | None = None,
        run_manifest: dict[str, Any] | None = None,
    ) -> None:
        self.run_manifest = run_manifest
        self.annotation_validator = annotation_validator
        self.budget_runtime = budget_runtime
        self._generator = generator
        self._judge_model = judge_model
        self._checkpoints = checkpoints

    async def evaluate(
        self,
        idx: int,
        question: dict[str, Any],
        run_idx: int,
        ctx: RagContext,
        checkpoint_path: Path | None = None,
    ) -> dict[str, Any]:

        saved = await load_stages(self._checkpoints, checkpoint_path)
        context_token = judge_context.set(
            {
                "case_id": question.get("case_id") or uuid4().hex,
                "answer_type": answer_type(question.get("annotations")),
                "dataset": question.get("dataset"),
                "variant_of": question.get("variant_of"),
                "question_id": question.get("id", idx),
                "run": run_idx,
                "checkpoint": str(checkpoint_path) if checkpoint_path else None,
            }
        )
        records = saved.get("judge_usage", judge_usage.get())

        if records is None:
            records = []

        token = judge_usage.set(records)
        try:
            result = await self.evaluate_case(idx, question, run_idx, ctx, checkpoint_path)
            result["judge_input_tokens"] = sum(r.get("input_tokens") or 0 for r in records)
            result["judge_output_tokens"] = sum(r.get("output_tokens") or 0 for r in records)
            result["judge_calls"] = len(records)
            all_records = (result.get("rag_usage") or []) + records
            result["judge_usage"] = records
            result["total_usage"] = summarize_usage(all_records)
            result["total_cost_usd"] = (
                result["total_usage"]["cost_usd"] if result.get("rag_usage") is not None else None
            )
            result["cost_usd"] = result["total_cost_usd"]
            result["usage_scope"] = (
                "observed RAG model calls plus judge responses, including validation retries"
            )
            return result
        finally:
            judge_usage.reset(token)
            judge_context.reset(context_token)

            if checkpoint_path:
                saved = await load_stages(self._checkpoints, checkpoint_path)
                saved["judge_usage"] = records
                await save_stages(self._checkpoints, checkpoint_path, saved)

    async def evaluate_case(
        self,
        idx: int,
        question: dict[str, Any],
        run_idx: int,
        ctx: RagContext,
        checkpoint_path: Path | None = None,
    ) -> dict[str, Any]:

        annotations = normalize_annotations(validate_annotations(question.get("annotations")))
        started = time.monotonic()
        stages = await load_stages(self._checkpoints, checkpoint_path)

        if refresh_judge_rubric(stages):
            await save_stages(self._checkpoints, checkpoint_path, stages)

        ctx = question_context(question, ctx)
        validation = stages.get("annotation_validation")

        if validation is None:
            validation = (
                await self.annotation_validator.validate(annotations, ctx)
                if self.annotation_validator
                else {"status": "unvalidated", "reason": "validator_unavailable"}
            )
            stages["annotation_validation"] = validation
            await save_stages(self._checkpoints, checkpoint_path, stages)

        generated = await self.generate_answer(question, ctx, stages, checkpoint_path)

        lock = asyncio.Lock()

        async def save_stage(name: str, value: dict[str, Any]) -> None:
            async with lock:
                stages[name] = value

                stages["judge_usage"] = list(judge_usage.get() or [])
                await save_stages(self._checkpoints, checkpoint_path, stages)

        evaluation_annotations = report_annotations(annotations, validation)
        judging_question = (
            question
            if validation["status"] != "invalid_annotation"
            else {**question, "annotations": None, "expected_answer": None}
        )
        generator_metrics, context_metrics, judge_metrics = await judge_validated(
            judging_question,
            generated,
            evaluation_annotations,
            stages,
            save_stage,
            policy={
                "judge_model": self._judge_model,
                "runtime": self.budget_runtime,
                "annotation_status": validation["status"],
            },
        )
        judgement = CaseJudgement(evaluation_annotations, generator_metrics, context_metrics, judge_metrics)
        return self.report_case(question, generated, validation, judgement, CaseRun(idx, run_idx, started))

    async def generate_answer(
        self, question: dict[str, Any], ctx: RagContext, stages: dict[str, Any], checkpoint_path: Path | None
    ) -> BenchmarkAnswer:
        saved_answer = stages.get("generated")

        if saved_answer is not None:
            saved_answer = dict(saved_answer)
            evidence = saved_answer.get("evidence")
            saved_answer["evidence"] = BenchmarkEvidence(**evidence) if evidence is not None else None
            generated = BenchmarkAnswer(**saved_answer)
        else:
            try:
                # References are evaluator-only; even custom generators receive no labels.
                generation_question = {
                    key: question[key] for key in ("question", "history", "as_of_date") if key in question
                }
                generated = await self._generator.generate(generation_question, ctx)
            except BaseException as error:
                failure = getattr(error, "benchmark_evidence", None)

                if failure is not None:
                    stages.setdefault("failures", []).append(asdict(failure))
                    await save_stages(self._checkpoints, checkpoint_path, stages)

                raise
            stages["generated"] = asdict(generated)
            await save_stages(self._checkpoints, checkpoint_path, stages)

        return generated

    def report_case(
        self,
        question: dict[str, Any],
        generated: BenchmarkAnswer,
        validation: dict[str, Any],
        judgement: CaseJudgement,
        run: CaseRun,
    ) -> dict[str, Any]:
        evaluation_annotations = judgement.annotations
        generator_metrics = judgement.generator_metrics
        context_metrics = judgement.context_metrics
        judge_metrics = judgement.judge_metrics
        evidence = generated.evidence
        evidence_metrics, diagnostics = evaluate_evidence(
            evaluation_annotations,
            evidence.retrieved if evidence else None,
            evidence.selected if evidence else None,
            evidence.context if evidence else None,
            generated.answer,
            judge_scores=judge_metrics["scores"],
        )
        retrieval = evaluate_retrieval(
            question if validation["status"] != "invalid_annotation" else {"annotations": None},
            evidence.retrieved if evidence else None,
        )
        saved_evidence: dict[str, Any] = asdict(evidence) if evidence else {"schema_version": 1}
        saved_evidence["annotation_validation"] = validation
        stage_evaluation = evaluate_stage_evidence(evaluation_annotations, saved_evidence, generated.answer)
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

        if (evidence is None or evidence.retrieved is None) and answer_type(
            question.get("annotations")
        ) not in ABSTENTION_TYPES:
            retriever_metrics["source_hit_rate"] = generated.retriever_metrics.get("hit_rate")
            retriever_metrics["source_mrr"] = generated.retriever_metrics.get("mrr")

        # Legacy file metrics remain explicitly named diagnostics, never an
        # alternative result when annotated evidence is unavailable.
        retriever_metrics.setdefault("avg_similarity", 0.0)
        retriever_metrics.setdefault("retrieved_sources", [])
        diagnostics["judge"] = judge_metrics["details"]
        cost = metrics.estimate_cost_usd(
            settings.openrouter_model, generated.input_tokens or 0, generated.output_tokens or 0
        )
        return {
            "id": question.get("id", str(run.idx)),
            "question": question["question"],
            "case_id": question.get("case_id"),
            "variant_of": question.get("variant_of"),
            "question_style": question.get("question_style"),
            "answer_type": answer_type(question.get("annotations")),
            "tags": question.get("tags") or [],
            "dataset": question.get("dataset"),
            "annotation_validation": validation,
            "run_manifest": self.run_manifest,
            "stage_evaluation": stage_evaluation,
            "diagnostic_card": diagnostic_card(
                {
                    "id": question.get("id", str(run.idx)),
                    "question": question["question"],
                    "stage_evaluation": stage_evaluation,
                }
            ),
            "answer": generated.answer,
            "expected_answer": question.get("expected_answer")
            if validation["status"] != "invalid_annotation"
            else None,
            "source_hint": question.get("source_hint")
            if validation["status"] != "invalid_annotation"
            else None,
            "retriever_metrics": retriever_metrics,
            "annotations": evaluation_annotations,
            "evidence_metrics": evidence_metrics,
            "evidence_diagnostics": diagnostics,
            "evidence": {
                **saved_evidence,
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
            "latency_sec": round(time.monotonic() - run.started, 2),
            "rag_latency_sec": generated.rag_latency_sec,
            "rag_usage": generated.llm_usage,
            "rag_cost_usd": summarize_usage(generated.llm_usage)["cost_usd"]
            if generated.llm_usage is not None
            else None,
            "input_tokens": generated.input_tokens,
            "output_tokens": generated.output_tokens,
            "estimated_generation_cost_usd": round(cost, 6),
            "generation_sec": generated.generation_sec,
            "ttft_sec": generated.ttft_sec,
            "breadth": generated.breadth,
            "domain": generated.domain,
            "run": run.run_idx,
        }


async def judge_validated(
    question: dict[str, Any],
    generated: BenchmarkAnswer,
    annotations: dict[str, Any] | None,
    stages: dict[str, Any],
    save_stage: SaveStage,
    *,
    policy: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if policy["annotation_status"] == "invalid_annotation":
        return (
            dict.fromkeys(("faithfulness", "relevancy", "correctness")),
            dict.fromkeys(("context_precision", "context_recall")),
            {"scores": {}, "details": {"annotation_status": "invalid_annotation"}},
        )

    return await judge_generated(
        question,
        generated,
        annotations,
        stages,
        save_stage,
        judge_model=policy["judge_model"],
        runtime=policy["runtime"],
    )
