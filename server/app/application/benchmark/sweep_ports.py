"""Sweep contracts, cancellation and progress."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
import inspect
import logging
from typing import TYPE_CHECKING, Any, Protocol


if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from application.benchmark.sweep_dto import CandidateCache, DenseCache, RerankerCache, SparseCache


logger = logging.getLogger("default")
ScoreConfig = Callable[[dict[str, Any]], dict[str, Any]]
ProgressCallback = Callable[[int, int, dict[str, Any] | None], Awaitable[None] | None]
ShouldCancel = Callable[[], Awaitable[bool]]


async def report_progress(
    callback: ProgressCallback | None, evaluated: int, total: int, result: dict[str, Any] | None
) -> None:
    if callback is not None:
        pending = callback(evaluated, total, result)

        if inspect.isawaitable(pending):
            await pending


class SweepCancelled(Exception):  # noqa: N818 - existing public cancellation contract.
    """Cancellation requested between sweep evaluations."""


async def check_cancelled(should_cancel: ShouldCancel | None, where: str) -> None:
    if should_cancel is not None and await should_cancel():
        logger.info("Sweep cancelled at %s", where)
        raise SweepCancelled(where)


class SweepSearchStrategy(Protocol):
    def max_fetch_k(self, default: int) -> int: ...

    async def evaluate(
        self, score: ScoreConfig, progress: ProgressCallback | None, should_cancel: ShouldCancel | None
    ) -> list[dict[str, Any]]: ...


class SweepSettingsPort(Protocol):
    @property
    def fetch_k(self) -> int: ...
    @property
    def top_k(self) -> int: ...
    def override(self, config: dict[str, Any]) -> AbstractContextManager[None]: ...
    def snapshot(self) -> dict[str, Any]: ...


class SweepDataPort(Protocol):
    async def load_questions(self, dataset: str) -> list[dict[str, Any]]: ...
    async def cache_candidates(
        self, questions: list[dict[str, Any]], max_fetch_k: int
    ) -> tuple[DenseCache, SparseCache, CandidateCache]: ...
    async def cache_reranker_scores(
        self,
        questions: list[dict[str, Any]],
        dense: DenseCache,
        sparse: SparseCache,
        candidates: CandidateCache,
        should_cancel: ShouldCancel | None,
    ) -> dict[str, Any]: ...


class SweepScoringPort(Protocol):
    def __call__(  # noqa: PLR0913, PLR0917 - scoring cached candidates needs all cache inputs.
        self,
        config: dict[str, Any],
        questions: list[dict[str, Any]],
        dense_cache: DenseCache,
        sparse_cache: SparseCache,
        all_candidates: CandidateCache,
        weights: dict[str, Any],
        *,
        rerank_scores: RerankerCache,
        defaults: dict[str, Any],
    ) -> dict[str, Any]: ...


SweepCorpus = Callable[[], AbstractAsyncContextManager[dict[str, Any]]]
