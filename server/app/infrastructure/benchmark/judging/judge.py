"""Benchmark judge — LLM-as-judge scoring with structured output."""

from infrastructure.benchmark.judging.clients import get_judge_model, get_judge_client
from infrastructure.benchmark.judging.prompts import FAITHFULNESS_PROMPT, RELEVANCY_PROMPT, CORRECTNESS_PROMPT

import asyncio
import logging
import json

from instructor.core.exceptions import IncompleteOutputException
from openai import APIConnectionError, RateLimitError, InternalServerError
from pydantic import ValidationError
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from config import settings
from domain.runtime.enums import LLMProvider

from infrastructure.ml.clients.llm_schemas import JudgeScore
from infrastructure.ml.clients.client_lease import sync_client_operation


logger = logging.getLogger("default")

JUDGE_OUTPUT_TOKENS = 2048
JUDGE_EXPANDED_OUTPUT_TOKENS = 8192


def judge_with_structured_output(
    client,
    prompt: str,
    model: str,
) -> JudgeScore:
    """Call judge LLM with structured output via instructor.

    Uses two attempts for transient or validation failures; HTTP 400 is terminal.
    Truncation permits one additional call with a larger output budget and no retries.
    """
    with sync_client_operation(client):
        initial_budget = settings.benchmark_judge_initial_tokens
        retry_budget = settings.benchmark_judge_retry_tokens

        for attempt, budget in enumerate((initial_budget, retry_budget)):
            try:
                return client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    response_model=JudgeScore,
                    max_tokens=budget,
                    timeout=settings.llm_auxiliary_timeout,
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

        raise RuntimeError("Judge output budget exhausted")


# ---------------------------------------------------------------------------
# Judge scoring
# ---------------------------------------------------------------------------


def judge_answer(
    question: str,
    answer: str,
    context: str,
    expected_answer: str | None = None,
    judge_llm=None,
) -> dict:
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
            client = get_judge_client(model if settings.llm_provider == LLMProvider.OLLAMA else "")
            result = judge_with_structured_output(client, prompt, model)
            scores[key] = max(0.0, min(10.0, result.score))
            scores[f"{key}_reason"] = result.reason
        except Exception as exc:
            logger.warning("Judge structured output failed for %s: %s", key, exc)
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
    judge_llm=None,
    *,
    judge_model: str | None = None,
    saved_metrics: dict | None = None,
    metric_callback=None,
) -> dict:
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
        client = get_judge_client(model if settings.llm_provider == LLMProvider.OLLAMA else "")
        with sync_client_operation(client):
            return await asyncio.wait_for(
                asyncio.to_thread(judge_with_structured_output, client, prompt, model),
                timeout=settings.llm_auxiliary_timeout * 3 + 3,
            )

    scores = dict(saved_metrics or {})

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
            logger.warning("Judge failed for %s: %s", key, exc)
            scores[key] = None
            scores[f"{key}_error"] = str(exc)
            scores[f"{key}_reason"] = f"[Error: {exc}]"

        if metric_callback is not None:
            await metric_callback(dict(scores))

    if "correctness" not in scores:
        scores["correctness"] = None
        scores["correctness_reason"] = "Эталонный ответ не задан"

    return scores
