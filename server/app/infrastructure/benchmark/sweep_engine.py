"""Coordinate dataset preparation, retrieval search strategies and full evaluation."""

from __future__ import annotations

import logging
from config import get_setting, settings
from pathlib import Path

from application.ports.benchmark_checkpoints import BenchmarkCheckpoints
from infrastructure.benchmark.checkpoint import (
    fingerprint,
    FileBenchmarkCheckpoints,
    import_legacy_configuration,
)
from collections.abc import Callable
from typing import Protocol

from application.ports.unit_of_work_factory import UnitOfWorkFactory
from domain.entities.benchmark_sweep import BenchmarkSweep
from domain.value_objects.sweep_evaluation_mode import SweepEvaluationMode, validate_sweep_mode
from domain.value_objects.benchmark_scoring import validate_objective_weights

from infrastructure.benchmark.sweep_data import SweepDataSource
from infrastructure.benchmark.sweep_full_evaluation import SweepFullEvaluator
from infrastructure.benchmark.sweep_scoring import has_retrieval_labels, score_config_cheap
from infrastructure.benchmark.sweep_settings import LiveSweepSettings, SweepSettingsPort
from infrastructure.benchmark.sweep_strategies import (
    ProgressCallback,
    ShouldCancel,
    SweepSearchStrategy,
    check_cancelled,
    default_strategy_factories,
    report_progress,
)
from infrastructure.benchmark.sweep_strategies import (
    SweepCancelled as SweepCancelled,
)

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
        questions: list[dict],
        out_dir: str,
        top_k: int,
        judge_model: str,
        *,
        resume: bool = False,
        checkpoints: BenchmarkCheckpoints | None = None,
        checkpoint_prefix: str | None = None,
    ) -> dict: ...


class SweepEngine:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        benchmark_service: SweepBenchmarkPort | None = None,
        ml_clients=None,
        *,
        runtime: SweepSettingsPort | None = None,
        strategies: dict[str, Callable[[dict], SweepSearchStrategy]] | None = None,
        checkpoint_factory: Callable[[int, Path], BenchmarkCheckpoints] | None = None,
    ) -> None:
        self._runtime = runtime if runtime is not None else LiveSweepSettings()
        self._checkpoint_factory = checkpoint_factory
        self._benchmark_service = benchmark_service
        self._data = SweepDataSource(uow_factory, ml_clients)
        self._full = SweepFullEvaluator(self._runtime)
        self._strategies = strategies if strategies is not None else default_strategy_factories()

    async def run_sweep(
        self,
        sweep: BenchmarkSweep,
        judge_model: str | None = None,
        progress_callback: ProgressCallback | None = None,
        should_cancel: ShouldCancel | None = None,
    ) -> list[dict]:
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
        self._checkpoint_root = (
            Path(self._runtime.results_path) / "sweeps" / str(sweep.id) if sweep.id else None
        )
        self._checkpoints = (
            self._checkpoint_factory(sweep.id, self._checkpoint_root)
            if self._checkpoint_factory and sweep.id
            else FileBenchmarkCheckpoints(self._checkpoint_root)
            if self._checkpoint_root
            else None
        )
        dataset_questions = await self._checkpoints.load("dataset.json") if self._checkpoints else None
        if dataset_questions is None:
            dataset_questions = await self.load_questions(sweep.dataset)
            if self._checkpoints:
                await self._checkpoints.save("dataset.json", dataset_questions)
        questions = [q for q in dataset_questions if has_retrieval_labels(q)]
        runtime_snapshot = await self.load_runtime_snapshot()
        self._baseline_config = {
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
        results = await self._checkpoints.load(phase_a_key) if self._checkpoints else None
        if results is None:
            results = await self.evaluate_phase_a(
                sweep, questions, strategy, progress_callback, should_cancel
            )
            if self._checkpoints:
                await self._checkpoints.save(phase_a_key, results)
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

    async def evaluate_phase_a(
        self,
        sweep: BenchmarkSweep,
        questions: list[dict],
        strategy: SweepSearchStrategy,
        progress_callback: ProgressCallback | None,
        should_cancel: ShouldCancel | None,
    ) -> list[dict]:
        expensive = set(sweep.search_space) & EXPENSIVE_PARAMS
        if expensive:
            logger.warning("Expensive params skipped in retrieval scoring: %s", expensive)
        dense, sparse, candidates = await self.cache_candidates(
            questions, strategy.max_fetch_k(self._runtime.fetch_k)
        )
        rerank_scores = await self._data.cache_reranker_scores(
            questions, dense, sparse, candidates, should_cancel
        )

        def score(config: dict) -> dict:
            cheap = {k: v for k, v in config.items() if k in CHEAP_PARAMS}
            return self.score_config_cheap(
                cheap, questions, dense, sparse, candidates, sweep.objective_weights, rerank_scores
            )

        return await strategy.evaluate(score, progress_callback, should_cancel)

    async def check_cancelled(self, should_cancel: ShouldCancel | None, where: str) -> None:
        await check_cancelled(should_cancel, where)

    async def load_questions(self, dataset: str) -> list[dict]:
        return await self._data.load_questions(dataset)

    async def cache_candidates(self, questions: list[dict], max_fetch_k: int) -> tuple[dict, dict, dict]:
        return await self._data.cache_candidates(questions, max_fetch_k)

    def score_config_cheap(
        self,
        config: dict,
        questions: list[dict],
        dense_cache: dict,
        sparse_cache: dict,
        all_candidates: dict,
        weights: dict,
        rerank_scores: dict,
    ) -> dict:
        return score_config_cheap(
            config,
            questions,
            dense_cache,
            sparse_cache,
            all_candidates,
            weights,
            rerank_scores=rerank_scores,
        )

    async def run_phase_b(
        self,
        results: list[dict],
        top_n_llm: int,
        judge_model: str,
        questions: list[dict],
        weights: dict,
        should_cancel: ShouldCancel | None = None,
        *,
        evaluation_mode: SweepEvaluationMode = SweepEvaluationMode.FAST,
        progress_callback: ProgressCallback | None = None,
    ) -> list[dict]:
        self._legacy_single_config = (
            len(results) == 1
            if evaluation_mode == SweepEvaluationMode.FULL
            else top_n_llm == 1 or len(results) == 1
        )
        with self._runtime.override(getattr(self, "_baseline_config", {})):
            return await self._full.evaluate(
                results,
                top_n_llm,
                judge_model,
                questions,
                weights,
                should_cancel,
                self.run_full_benchmark,
                evaluation_mode=evaluation_mode,
                progress_callback=progress_callback,
            )

    async def run_full_benchmark(self, questions: list[dict], judge_model: str) -> dict:
        if self._benchmark_service is None:
            raise RuntimeError("A benchmark service must be injected for full sweep evaluation")
        checkpoint_root = getattr(self, "_checkpoint_root", None)
        out_dir = self._runtime.results_path
        if checkpoint_root:
            # The active override identifies this configuration; runner also
            # fingerprints questions, model and effective runtime settings.
            out_dir = str(
                checkpoint_root
                / fingerprint(
                    {
                        "top_k": self._runtime.top_k,
                        "fetch_k": self._runtime.fetch_k,
                        "judge": judge_model,
                        "settings": {
                            name: get_setting(name)
                            for name in type(settings).model_fields
                            if not any(part in name for part in ("key", "password", "secret", "url", "dir"))
                            and name
                            not in {
                                "benchmark_judge_initial_tokens",
                                "benchmark_judge_retry_tokens",
                                "benchmark_judge_grouped_enabled",
                                "benchmark_judge_max_concurrent",
                            }
                        },
                    }
                )
            )
        extra = {}
        if checkpoint_root:
            config_identity = fingerprint(
                {
                    "top_k": self._runtime.top_k,
                    "fetch_k": self._runtime.fetch_k,
                    "judge": judge_model,
                    **{name: get_setting(name) for name in CHEAP_PARAMS - {"top_k", "fetch_k"}},
                }
            )
            # Freeze the namespace on the first attempt. Operational setting
            # changes and worker restarts must not invalidate completed work.
            identity_key = f"config-{config_identity}.json"
            prefix = await self._checkpoints.load(identity_key)
            if prefix is None:
                prefix = fingerprint(
                    {"config": config_identity, "questions": questions, "judge": judge_model}
                )
                await self._checkpoints.save(identity_key, prefix)
                if self._checkpoint_factory:
                    await import_legacy_configuration(
                        self._checkpoints,
                        Path(out_dir),
                        prefix,
                        single_config_root=checkpoint_root if self._legacy_single_config else None,
                    )
            extra = {"resume": True, "checkpoints": self._checkpoints, "checkpoint_prefix": prefix}
        return await self._benchmark_service.run(
            questions=questions,
            out_dir=out_dir,
            top_k=self._runtime.top_k,
            judge_model=judge_model,
            **extra,
        )

    async def load_runtime_snapshot(self) -> dict:
        runtime_snapshot = {
            name: get_setting(name)
            for name in type(settings).model_fields
            if not any(part in name for part in ("key", "password", "secret", "url", "dir"))
            and name
            not in {
                "benchmark_judge_initial_tokens",
                "benchmark_judge_retry_tokens",
                "benchmark_judge_grouped_enabled",
                "benchmark_judge_max_concurrent",
            }
        }
        if self._checkpoints:
            saved_runtime = await self._checkpoints.load("runtime.json")
            if saved_runtime is None:
                await self._checkpoints.save("runtime.json", runtime_snapshot)
            else:
                runtime_snapshot = saved_runtime
        return runtime_snapshot
