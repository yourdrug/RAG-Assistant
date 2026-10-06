"""Search strategies with cancellation and progress owned by the event loop."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Protocol

from domain.value_objects.benchmark_strategy import BenchmarkStrategy

from infrastructure.benchmark.sweep_scoring import generate_grid_points, generate_random_points

logger = logging.getLogger("default")
ScoreConfig = Callable[[dict], dict]
ProgressCallback = Callable[[int, int, dict | None], None]
ShouldCancel = Callable[[], Awaitable[bool]]


class SweepCancelled(Exception):
    """Cancellation requested between sweep evaluations."""


async def check_cancelled(should_cancel: ShouldCancel | None, where: str) -> None:
    if should_cancel is not None and await should_cancel():
        logger.info("Sweep cancelled at %s", where)
        raise SweepCancelled(where)


class SweepSearchStrategy(Protocol):
    def max_fetch_k(self, default: int) -> int: ...

    async def evaluate(
        self, score: ScoreConfig, progress: ProgressCallback | None, should_cancel: ShouldCancel | None
    ) -> list[dict]: ...


class EnumeratedSweepStrategy:
    def __init__(self, points: list[dict]) -> None:
        self._points = points

    def max_fetch_k(self, default: int) -> int:
        return max((p.get("fetch_k", default) for p in self._points), default=default)

    async def evaluate(
        self, score: ScoreConfig, progress: ProgressCallback | None, should_cancel: ShouldCancel | None
    ) -> list[dict]:
        results = []
        for idx, point in enumerate(self._points, 1):
            await check_cancelled(should_cancel, f"phase A config {idx}")
            result = await asyncio.to_thread(score, point)
            result["config"] = point
            results.append(result)
            if progress is not None:
                progress(idx, len(self._points), result)
        results.sort(key=lambda r: r.get("composite_score", 0), reverse=True)
        return results


class OptunaSweepStrategy:
    def __init__(self, search_space: dict) -> None:
        self._space = search_space
        self._n_trials = search_space.get("_n_trials", 50)
        if self._n_trials < 1:
            raise ValueError("_n_trials must be at least 1")

    def max_fetch_k(self, default: int) -> int:
        spec = self._space.get("fetch_k", {})
        if "values" in spec:
            return max(spec["values"], default=default)
        return spec.get("max", spec.get("default", default))

    def sample_config(self, trial) -> dict:
        config = {}
        for param, spec in self._space.items():
            if param.startswith("_"):
                continue
            if "values" in spec:
                config[param] = trial.suggest_categorical(param, spec["values"])
            elif "min" in spec and "max" in spec:
                step = spec.get("step")
                is_float = any(isinstance(v, float) for v in (spec["min"], spec["max"], step))
                if is_float:
                    config[param] = trial.suggest_float(param, spec["min"], spec["max"], step=step)
                else:
                    config[param] = trial.suggest_int(
                        param, int(spec["min"]), int(spec["max"]), step=1 if step is None else step
                    )
            else:
                config[param] = spec.get("default", 0)
        return config

    async def evaluate(
        self, score: ScoreConfig, progress: ProgressCallback | None, should_cancel: ShouldCancel | None
    ) -> list[dict]:
        import optuna

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        results = []

        def objective(trial):
            config = self.sample_config(trial)
            result = score(config)
            result["config"] = config
            results.append(result)
            return result.get("composite_score", 0.0)

        for evaluated in range(1, self._n_trials + 1):
            await check_cancelled(should_cancel, f"Optuna trial {evaluated}")
            await asyncio.to_thread(study.optimize, objective, n_trials=1)
            if progress is not None:
                progress(evaluated, self._n_trials, results[-1])
        await check_cancelled(should_cancel, "after Optuna trials")
        results.sort(key=lambda r: r.get("composite_score", 0), reverse=True)
        return results


def default_strategy_factories() -> dict[str, Callable[[dict], SweepSearchStrategy]]:
    return {
        BenchmarkStrategy.GRID.value: lambda space: EnumeratedSweepStrategy(generate_grid_points(space)),
        BenchmarkStrategy.RANDOM.value: lambda space: EnumeratedSweepStrategy(
            generate_random_points(space, space.get("_n_random", 50))
        ),
        BenchmarkStrategy.OPTUNA_TPE.value: OptunaSweepStrategy,
        # Keep previously persisted sweeps executable without a data migration.
        BenchmarkStrategy.SUCCESSIVE_HALVING.value: OptunaSweepStrategy,
    }
