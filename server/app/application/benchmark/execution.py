"""Schedule benchmark cases, repetitions and checkpoint recovery through ports."""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass
import logging
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any, Protocol, cast

from application.benchmark.identity import fingerprint
from domain.benchmark.judge import result_judge_errors
from domain.benchmark.question import validate_annotations
from domain.identity.roles import UserKind, UserRole
from domain.identity.user_context import UserContext
from domain.rag.context import RagContext


if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractAsyncContextManager

    from application.benchmark.ports import BenchmarkCheckpoints


logger = logging.getLogger(__name__)


class BenchmarkCasePort(Protocol):
    async def evaluate(
        self,
        idx: int,
        question: dict[str, Any],
        run_idx: int,
        ctx: RagContext,
        checkpoint_path: Path | None = None,
    ) -> dict[str, Any]: ...


class BenchmarkProgress(Protocol):
    def log_question_result(
        self, idx: int, total: int, question: dict[str, Any], result: dict[str, Any]
    ) -> None: ...
    def log_summary(self, results: list[dict[str, Any]], total_time: float) -> None: ...


@dataclass(frozen=True)
class BenchmarkExecutionScope:
    evaluator: BenchmarkCasePort
    corpus_metadata: dict[str, Any]


def validate_questions(questions: object) -> None:
    """Validate database cases before starting expensive evaluation."""
    if not isinstance(questions, list):
        message = "benchmark questions must be a list"
        raise ValueError(message)  # noqa: TRY004 — preserve benchmark validation contract

    if not questions:
        message = "benchmark requires at least one active question"
        raise ValueError(message)

    for question in questions:
        if not isinstance(question, dict) or not isinstance(question.get("question"), str):
            message = "each benchmark case requires a question string"
            raise ValueError(message)  # noqa: TRY004 — preserve benchmark validation contract

        validate_annotations(question.get("annotations"))


async def prepare_checkpoint_corpus(
    store: BenchmarkCheckpoints | None, checkpoint_dir: Path | None, metadata: dict[str, Any]
) -> Path | None:
    if store is None or checkpoint_dir is None or "corpus_namespace" not in metadata:
        return checkpoint_dir

    corpus_dir = checkpoint_dir / str(metadata["corpus_namespace"])
    corpus_key = str(corpus_dir / "corpus.json")
    saved_corpus = await store.load(corpus_key)

    if saved_corpus is not None and saved_corpus != metadata:
        message = "Benchmark corpus changed; start a new run"
        raise ValueError(message)

    await store.save(corpus_key, metadata)
    return corpus_dir


async def execute_benchmark(  # noqa: C901, PLR0913, PLR0915, PLR0917 — preserve scheduling and recovery together.
    questions: list[dict[str, Any]],
    top_k: int,
    judge_model: str,
    max_concurrent: int = 1,
    seed: int | None = None,
    n_runs: int = 1,
    *,
    scope: Callable[[], AbstractAsyncContextManager[BenchmarkExecutionScope]],
    reporter: BenchmarkProgress,
    settings_identity: str,
    rubric_version: str,
    fetch_k: int | None = None,
    resume: bool = False,
    checkpoints: BenchmarkCheckpoints | None = None,
    checkpoint_prefix: str | None = None,
) -> list[dict[str, Any]]:
    """Schedule benchmark cases, aggregate each run and persist the results."""
    if resume and checkpoints is None:
        message = "A database checkpoint store is required for resume"
        raise ValueError(message)

    validate_questions(questions)

    if n_runs < 1:
        message = "n_runs must be positive"
        raise ValueError(message)

    if max_concurrent < 1:
        message = "max_concurrent must be positive"
        raise ValueError(message)

    semaphore = asyncio.Semaphore(max_concurrent)
    completed = 0
    ctx = RagContext(user=UserContext(user_id=0, user_kind=UserKind.INTERNAL, user_role=UserRole.ADMIN))
    stack = AsyncExitStack()
    try:
        execution = await stack.enter_async_context(scope())
        # Store only a hash of settings: credentials must never enter report files.
        identity = fingerprint(
            {
                "corpus": execution.corpus_metadata,
                "questions": questions,
                "top_k": top_k,
                "fetch_k": fetch_k,
                "judge": judge_model,
                "seed": seed,
                "runs": n_runs,
                "settings": settings_identity,
            }
        )
        checkpoint_dir = Path("checkpoints") / (checkpoint_prefix or identity) if resume else None
        store = checkpoints
        checkpoint_dir = await prepare_checkpoint_corpus(store, checkpoint_dir, execution.corpus_metadata)
        evaluator = execution.evaluator
        all_results = []

        async def process_question(idx: int, question: dict[str, Any], run_idx: int) -> dict[str, Any]:
            nonlocal completed
            async with semaphore:
                logger.info(
                    "Benchmark question %s started (run %d, slot limit %d)",
                    question.get("id", idx),
                    run_idx,
                    max_concurrent,
                )
                path = checkpoint_dir / f"{run_idx}-{idx}.json" if checkpoint_dir else None
                cached = (
                    cast("dict[str, Any] | None", await store.load(str(path))) if path and store else None
                )

                if cached is not None and cached.get("judge_rubric_version") == rubric_version:
                    completed += 1
                    logger.info(
                        "Benchmark progress: %d/%d completed (restored question=%s)",
                        completed,
                        len(questions) * n_runs,
                        question.get("id", idx),
                    )
                    return cached

                result = await evaluator.evaluate(
                    idx,
                    question,
                    run_idx,
                    ctx,
                    **({"checkpoint_path": path.with_suffix(".stages.json")} if path else {}),
                )
                diagnostic_errors = bool(result_judge_errors(result))

                if path and store is not None and not diagnostic_errors:
                    await store.save(str(path), result)

                completed += 1
                logger.info(
                    "Benchmark progress: %d/%d completed; question=%s",
                    completed,
                    len(questions) * n_runs,
                    question.get("id", idx),
                )
                reporter.log_question_result(idx, len(questions), question, result)
                return result

        for run_idx in range(1, n_runs + 1):
            started = time.monotonic()
            # TaskGroup cancels sibling work on failure; gather previously left paid
            # requests running while the worker restarted the entire sweep.
            try:
                async with asyncio.TaskGroup() as group:
                    tasks = [
                        group.create_task(process_question(idx, q, run_idx))
                        for idx, q in enumerate(questions, 1)
                    ]
            except ExceptionGroup as exc:
                # Preserve the actionable provider error in the persisted job error.
                failure = exc.exceptions[0]

                while isinstance(failure, ExceptionGroup):
                    failure = failure.exceptions[0]

                raise failure from exc
            results = [task.result() for task in tasks]
            reporter.log_summary(list(results), time.monotonic() - started)
            all_results.extend(results)

        return all_results
    finally:
        await stack.aclose()
