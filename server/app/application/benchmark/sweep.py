"""Coordinate dataset preparation, retrieval search strategies and full evaluation."""

from __future__ import annotations
from application.benchmark.sweep_dto import DenseCache, SparseCache, CandidateCache, RerankerCache

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
import logging
from typing import Any, Protocol, TypedDict, cast

from application.benchmark import BenchmarkCheckpoints
from application.benchmark.identity import fingerprint
from application.benchmark.sweep_evaluation import SweepFullEvaluator
from application.benchmark.sweep_ports import (
    ProgressCallback,
    ShouldCancel,
    SweepCancelled as SweepCancelled,
    SweepCorpus,
    SweepDataPort,
    SweepScoringPort,
    SweepSearchStrategy,
    SweepSettingsPort,
    check_cancelled,
    report_progress,
)
from domain.benchmark.evaluation import has_retrieval_labels
from domain.benchmark.scoring import validate_objective_weights
from domain.benchmark.sweep import BenchmarkSweep, SweepEvaluationMode, validate_sweep_mode


logger = logging.getLogger("default")
CHEAP_PARAMS = frozenset(
    {
        "top_k",
        "fetch_k",
        "rrf_k",
        "dense_weight",
        "sparse_weight",
        "rerank_min_score",
        "rerank_score_gap_ratio",
    }
)
EXPENSIVE_PARAMS = frozenset({"chunk_size", "chunk_overlap"})


class SweepBenchmarkPort(Protocol):
    async def run(
        self,
        questions: list[dict[str, Any]],
        top_k: int,
        judge_model: str,
        *,
        resume: bool = False,
        checkpoints: BenchmarkCheckpoints | None = None,
        checkpoint_prefix: str | None = None,
    ) -> dict[str, Any]: ...


class SweepResumeOptions(TypedDict, total=False):
    resume: bool
    checkpoints: BenchmarkCheckpoints
    checkpoint_prefix: str


@dataclass
class SweepRun:
    corpus: dict[str, Any] | None = None
    checkpoints: BenchmarkCheckpoints | None = None
    baseline_config: dict[str, Any] = field(default_factory=dict[str, Any])


class SweepEngine:
    def __init__(
        self,
        *,
        data: SweepDataPort,
        runtime: SweepSettingsPort,
        full_evaluator: SweepFullEvaluator,
        scoring: SweepScoringPort,
        corpus: SweepCorpus,
        strategies: dict[str, Callable[[dict[str, Any]], SweepSearchStrategy]],
        benchmark_service: SweepBenchmarkPort | None = None,
        checkpoint_factory: Callable[[int], BenchmarkCheckpoints] | None = None,
    ) -> None:
        self._runtime = runtime
        self._checkpoint_factory = checkpoint_factory
        self._benchmark_service = benchmark_service
        self._data = data
        self._full = full_evaluator
        self._strategies = strategies
        self._scoring = scoring
        self._corpus = corpus

    async def run_sweep(
        self,
        sweep: BenchmarkSweep,
        judge_model: str | None = None,
        progress_callback: ProgressCallback | None = None,
        should_cancel: ShouldCancel | None = None,
    ) -> list[dict[str, Any]]:
        async with self._corpus() as corpus:
            run = SweepRun(corpus=corpus)
            mode = validate_sweep_mode(sweep.evaluation_mode, sweep.strategy)
            validate_objective_weights(
                sweep.objective_weights, require_retrieval=mode == SweepEvaluationMode.FAST
            )

            if mode == SweepEvaluationMode.FULL and not judge_model:
                raise ValueError("Full evaluation requires a judge model")

            unsupported = {
                key for key in sweep.search_space if not key.startswith("_") and key not in CHEAP_PARAMS
            }

            if mode == SweepEvaluationMode.FULL and unsupported:
                raise ValueError(
                    f"Full evaluation supports retrieval parameters only; unsupported: {sorted(unsupported)}"
                )

            try:
                factory = self._strategies[sweep.strategy]
            except KeyError as exc:
                raise ValueError(f"Unknown strategy: {sweep.strategy}") from exc
            strategy = factory(sweep.search_space)
            await self.check_cancelled(should_cancel, "before sweep preparation")
            run.checkpoints = self.create_checkpoints(sweep.id)
            dataset_questions = (
                cast("list[dict[str, Any]] | None", await run.checkpoints.load("dataset.json"))
                if run.checkpoints
                else None
            )

            if dataset_questions is None:
                dataset_questions = await self.load_questions(sweep.dataset)

                if run.checkpoints:
                    await run.checkpoints.save("dataset.json", dataset_questions)

            questions = [q for q in dataset_questions if has_retrieval_labels(q)]
            runtime_snapshot = await self.load_runtime_snapshot(run)
            run.baseline_config = {
                "top_k": runtime_snapshot["retriever_top_k"],
                "fetch_k": runtime_snapshot["retriever_fetch_k"],
                **{name: runtime_snapshot[name] for name in CHEAP_PARAMS - {"top_k", "fetch_k"}},
            }
            phase_a_identity = fingerprint(
                {
                    "questions": questions,
                    "space": sweep.search_space,
                    "strategy": sweep.strategy,
                    "weights": sweep.objective_weights,
                    "settings": runtime_snapshot,
                }
            )
            phase_a_key = f"phase-a-{phase_a_identity}.json"
            results = (
                cast("list[dict[str, Any]] | None", await run.checkpoints.load(phase_a_key))
                if run.checkpoints
                else None
            )

            if results is None:
                results = await self.evaluate_phase_a(
                    sweep, questions, strategy, progress_callback, should_cancel
                )

                if run.checkpoints:
                    await run.checkpoints.save(phase_a_key, results)
            else:
                logger.info("Sweep %s: restored phase A (%d configurations)", sweep.id, len(results))
                await report_progress(
                    progress_callback, len(results), len(results), {"phase": "retrieval", "resumed": True}
                )

            if judge_model and (mode == SweepEvaluationMode.FULL or sweep.top_n_llm > 0):
                results = await self.run_phase_b(
                    results,
                    sweep.top_n_llm,
                    judge_model,
                    dataset_questions,
                    sweep.objective_weights,
                    should_cancel,
                    run=run,
                    evaluation_mode=mode,
                    progress_callback=progress_callback,
                )
            else:
                for result in results:
                    result["evaluation_mode"] = mode.value
                    result["evaluated_config_count"] = 0
                    result["scored_config_count"] = 0
                    result["search_config_count"] = len(results)

            return results

    def create_checkpoints(self, sweep_id: int | None) -> BenchmarkCheckpoints | None:
        if self._checkpoint_factory and sweep_id:
            return self._checkpoint_factory(sweep_id)

        if sweep_id:
            raise RuntimeError("A database checkpoint store is required for web sweeps")

        return None

    async def evaluate_phase_a(
        self,
        sweep: BenchmarkSweep,
        questions: list[dict[str, Any]],
        strategy: SweepSearchStrategy,
        progress_callback: ProgressCallback | None,
        should_cancel: ShouldCancel | None,
    ) -> list[dict[str, Any]]:
        expensive = set(sweep.search_space) & EXPENSIVE_PARAMS

        if expensive:
            logger.warning("Expensive params skipped in retrieval scoring: %s", expensive)

        dense, sparse, candidates = await self.cache_candidates(
            questions, strategy.max_fetch_k(self._runtime.fetch_k)
        )
        rerank_scores = await self._data.cache_reranker_scores(
            questions, dense, sparse, candidates, should_cancel
        )

        def score(config: dict[str, Any]) -> dict[str, Any]:
            cheap = {k: v for k, v in config.items() if k in CHEAP_PARAMS}
            return self.score_config_cheap(
                cheap, questions, dense, sparse, candidates, sweep.objective_weights, rerank_scores
            )

        return await strategy.evaluate(score, progress_callback, should_cancel)

    async def check_cancelled(self, should_cancel: ShouldCancel | None, where: str) -> None:
        await check_cancelled(should_cancel, where)

    async def load_questions(self, dataset: str) -> list[dict[str, Any]]:
        return await self._data.load_questions(dataset)

    async def cache_candidates(
        self, questions: list[dict[str, Any]], max_fetch_k: int
    ) -> tuple[DenseCache, SparseCache, CandidateCache]:
        return await self._data.cache_candidates(questions, max_fetch_k)

    def score_config_cheap(
        self,
        config: dict[str, Any],
        questions: list[dict[str, Any]],
        dense_cache: DenseCache,
        sparse_cache: SparseCache,
        all_candidates: CandidateCache,
        weights: dict[str, Any],
        rerank_scores: RerankerCache,
    ) -> dict[str, Any]:
        return self._scoring(
            config,
            questions,
            dense_cache,
            sparse_cache,
            all_candidates,
            weights,
            rerank_scores=rerank_scores,
            defaults=self._runtime.snapshot(),
        )

    async def run_phase_b(
        self,
        results: list[dict[str, Any]],
        top_n_llm: int,
        judge_model: str,
        questions: list[dict[str, Any]],
        weights: dict[str, Any],
        should_cancel: ShouldCancel | None = None,
        *,
        run: SweepRun | None = None,
        evaluation_mode: SweepEvaluationMode = SweepEvaluationMode.FAST,
        progress_callback: ProgressCallback | None = None,
    ) -> list[dict[str, Any]]:
        run = run if run is not None else SweepRun()
        with self._runtime.override(run.baseline_config):
            return await self._full.evaluate(
                results,
                top_n_llm,
                judge_model,
                questions,
                weights,
                should_cancel,
                partial(self.run_full_benchmark, run=run),
                evaluation_mode=evaluation_mode,
                progress_callback=progress_callback,
            )

    async def run_full_benchmark(
        self, questions: list[dict[str, Any]], judge_model: str, *, run: SweepRun
    ) -> dict[str, Any]:
        if self._benchmark_service is None:
            raise RuntimeError("A benchmark service must be injected for full sweep evaluation")

        checkpoints = run.checkpoints
        extra: SweepResumeOptions = {}

        if checkpoints is not None:
            current_settings = self._runtime.snapshot()
            config_identity = fingerprint(
                {
                    "top_k": self._runtime.top_k,
                    "fetch_k": self._runtime.fetch_k,
                    "judge": judge_model,
                    **{name: current_settings[name] for name in CHEAP_PARAMS - {"top_k", "fetch_k"}},
                }
            )
            # Freeze the namespace on the first attempt. Operational setting
            # changes and worker restarts must not invalidate completed work.
            identity_key = f"config-{config_identity}.json"
            prefix = cast("str | None", await checkpoints.load(identity_key))

            if prefix is None:
                prefix = fingerprint(
                    {"config": config_identity, "questions": questions, "judge": judge_model}
                )
                await checkpoints.save(identity_key, prefix)

            extra = {"resume": True, "checkpoints": checkpoints, "checkpoint_prefix": prefix}

        return await self._benchmark_service.run(
            questions=questions,
            top_k=self._runtime.top_k,
            judge_model=judge_model,
            **extra,
        )

    async def load_runtime_snapshot(self, run: SweepRun) -> dict[str, Any]:
        runtime_snapshot = self._runtime.snapshot()
        runtime_snapshot.update(run.corpus or {})

        if run.checkpoints:
            saved_runtime = await run.checkpoints.load("runtime.json")

            if saved_runtime is None:
                await run.checkpoints.save("runtime.json", runtime_snapshot)
            else:
                if run.corpus is not None and cast("dict[str, Any]", saved_runtime).get(
                    "corpus_namespace"
                ) != runtime_snapshot.get("corpus_namespace"):
                    raise ValueError("Sweep corpus changed; start a new sweep")

                runtime_snapshot = cast("dict[str, Any]", saved_runtime)

        return runtime_snapshot
