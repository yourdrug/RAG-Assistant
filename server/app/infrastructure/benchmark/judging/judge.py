"""Benchmark judge — LLM-as-judge scoring with structured output."""

import asyncio
import json
import logging
from typing import Any, Unpack, cast

from instructor.core.exceptions import IncompleteOutputException
from openai import APIConnectionError, InternalServerError, RateLimitError
from pydantic import ValidationError
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from config import BenchmarkSection, LLMSection, process_section
from infrastructure.benchmark.judging.callbacks import AsyncMetricOptions
from infrastructure.benchmark.judging.clients import get_judge_client, get_judge_model
from infrastructure.benchmark.judging.prompts import CORRECTNESS_PROMPT, FAITHFULNESS_PROMPT, RELEVANCY_PROMPT
from infrastructure.budget.runtime import BudgetRuntime
from infrastructure.ml.clients.client_lease import sync_client_operation
from infrastructure.ml.clients.llm_schemas import JudgeScore
from infrastructure.ml.clients.managed_sync_instructor import ManagedSyncInstructor


__all__ = ["get_judge_client", "get_judge_model"]

logger = logging.getLogger("default")

JUDGE_OUTPUT_TOKENS = 2048
JUDGE_EXPANDED_OUTPUT_TOKENS = 8192


def judge_with_structured_output(
    client: ManagedSyncInstructor,
    prompt: str,
    model: str,
) -> JudgeScore:
    """Call judge LLM with structured output via instructor.

    Uses two attempts for transient or validation failures; HTTP 400 is terminal.
    Truncation permits one additional call with a larger output budget and no retries.
    """
    benchmark_settings = process_section(BenchmarkSection)

    with sync_client_operation(client):
        initial_budget = benchmark_settings.benchmark_judge_initial_tokens
        retry_budget = benchmark_settings.benchmark_judge_retry_tokens

        for attempt, budget in enumerate((initial_budget, retry_budget)):
            try:
                return cast(
                    "JudgeScore",
                    client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        response_model=JudgeScore,
                        max_tokens=budget,
                        timeout=process_section(LLMSection).llm_auxiliary_timeout,
                        max_retries=Retrying(
                            retry=retry_if_exception_type(
                                (
                                    APIConnectionError,
                                    RateLimitError,
                                    InternalServerError,
                                    ValidationError,
                                    json.JSONDecodeError,
                                )
                            ),
                            stop=stop_after_attempt(2 if attempt == 0 else 1),
                            wait=wait_exponential(multiplier=1, max=3),
                        ),
                    ),
                )
            except IncompleteOutputException:
                if attempt == 1:
                    raise

                logger.warning(
                    "Judge output truncated for model %s at %d tokens; retrying with %d tokens",
                    model,
                    budget,
                    retry_budget,
                )

        message = "Judge output budget exhausted"
        raise RuntimeError(message)


# ---------------------------------------------------------------------------
# Judge scoring
# ---------------------------------------------------------------------------


def judge_answer(
    question: str,
    answer: str,
    context: str,
    expected_answer: str | None = None,
    *,
    runtime: BudgetRuntime | None = None,
) -> dict[str, Any]:
    """Run judge LLM synchronously with structured output (instructor only)."""
    model = get_judge_model()
    prompts = {
        "faithfulness": FAITHFULNESS_PROMPT.format(context=context, question=question, answer=answer),
        "relevancy": RELEVANCY_PROMPT.format(question=question, answer=answer),
    }

    if expected_answer:
        prompts["correctness"] = CORRECTNESS_PROMPT.format(
            question=question, expected=expected_answer, answer=answer
        )

    scores: dict[str, float | str | None] = {}

    for key, prompt in prompts.items():
        try:
            client = get_judge_client(model, runtime=runtime)
            result = judge_with_structured_output(client, prompt, model)
            scores[key] = max(0.0, min(10.0, result.score))
            scores[f"{key}_reason"] = result.reason
        except Exception as exc:
            logger.warning(
                "Judge structured output failed for %s: %s",
                key,
                type(exc).__name__,
                exc_info=(RuntimeError, RuntimeError(type(exc).__name__), exc.__traceback__),
            )
            scores[key] = None
            scores[f"{key}_reason"] = f"[Ошибка вызова судьи: {exc}]"
            scores[f"{key}_error"] = str(exc)

    if "correctness" not in scores:
        scores["correctness"] = None
        scores["correctness_reason"] = "Эталонный ответ не задан"

    return scores


async def judge_answer_async(
    question: str,
    answer: str,
    context: str,
    expected_answer: str | None = None,
    *,
    judge_model: str | None = None,
    runtime: BudgetRuntime | None = None,
    **options: Unpack[AsyncMetricOptions],
) -> dict[str, Any]:
    """Judge answer quality with structured output (instructor + timeout)."""
    model = judge_model or get_judge_model()
    prompts = {
        "faithfulness": FAITHFULNESS_PROMPT.format(context=context, question=question, answer=answer),
        "relevancy": RELEVANCY_PROMPT.format(question=question, answer=answer),
    }

    if expected_answer:
        prompts["correctness"] = CORRECTNESS_PROMPT.format(
            question=question, expected=expected_answer, answer=answer
        )

    async def judge_one(prompt: str) -> JudgeScore:
        client = get_judge_client(model, runtime=runtime)
        with sync_client_operation(client):
            return await asyncio.wait_for(
                asyncio.to_thread(judge_with_structured_output, client, prompt, model),
                timeout=process_section(LLMSection).llm_auxiliary_timeout * 3 + 3,
            )

    scores = dict(options.get("saved_metrics") or {})
    metric_callback = options.get("metric_callback")

    for key, prompt in prompts.items():
        if scores.get(key) is not None:
            continue

        try:
            result = await judge_one(prompt)
            scores[key] = max(0.0, min(10.0, result.score))
            scores[f"{key}_reason"] = result.reason
            scores.pop(f"{key}_error", None)
        except Exception as exc:
            # Each metric is independent: provider failures are missing data,
            # never a numerical score, and must not discard sibling results.
            logger.warning(
                "Judge failed for %s: %s",
                key,
                type(exc).__name__,
                exc_info=(RuntimeError, RuntimeError(type(exc).__name__), exc.__traceback__),
            )
            scores[key] = None
            scores[f"{key}_error"] = str(exc)
            scores[f"{key}_reason"] = f"[Error: {exc}]"

        if metric_callback is not None:
            await metric_callback(dict(scores))

    if "correctness" not in scores:
        scores["correctness"] = None
        scores["correctness_reason"] = "Эталонный ответ не задан"

    return scores
