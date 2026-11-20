"""Shared base prompt and history budget for retrieval and generation.

The answer style must never decide which evidence reaches the model. Evidence selection,
history trimming and neighbour expansion therefore use one style-independent budget: the
common context window minus the largest output reserve of any style, measured against the
largest style prompt. The chosen style only picks its own prompt and output ceiling, both of
which fit inside that budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from domain.rag.breadth import AnswerStyle, answer_style
from domain.rag.exceptions import ContextBudgetExceededError
from infrastructure.ml.rag.evidence_focus import format_generation_messages
from infrastructure.ml.rag.prompt_budget import estimate_message_tokens
from infrastructure.ml.rag.rag_prompts import build_prompt, domain_prompt_addendum


if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.messages import BaseMessage
    from langchain_core.prompts import ChatPromptTemplate

    from domain.rag.settings import RagSettings
    from infrastructure.ml.rag.retrieval_work import RetrievalWork


@dataclass(frozen=True)
class GenerationBudget:
    # Prompt of the requested style: the one actually sent to the model.
    prompt: ChatPromptTemplate
    # The style prompt with the most tokens; selection counts against it so every style fits.
    selection_prompt: ChatPromptTemplate
    history: list[BaseMessage]
    num_ctx: int
    # Output ceiling of the requested style.
    num_predict: int
    # Input tokens available to prompt, history and evidence for every style.
    input_budget: int
    # Evidence tokens left after the selection prompt and history.
    remaining: int


def output_ceiling(rag: RagSettings, style: AnswerStyle) -> int:
    return rag.llm_num_predict_narrow if style == AnswerStyle.CONCISE else rag.llm_num_predict_broad


def selection_reserve(rag: RagSettings) -> int:
    """Output tokens reserved during evidence selection: the largest ceiling of any style."""
    return max(output_ceiling(rag, style) for style in AnswerStyle)


def prepare_generation_budget(
    state: RetrievalWork,
    count_tokens: Callable[[list[BaseMessage]], int] = estimate_message_tokens,
) -> GenerationBudget:
    registry = state.domain_registry
    keys = state.domain_keys
    addendum = domain_prompt_addendum(state.query_domain, state.ctx, keys, registry)
    prompts = {
        style: build_prompt(
            style,
            summary=state.ctx.summary,
            domain_addendum=addendum,
            enumerate_cases=state.enumerate_cases,
            question=state.question,
        )
        for style in AnswerStyle
    }
    style = answer_style(state.ctx.depth)
    num_ctx = state.rag.llm_num_ctx_broad
    input_budget = num_ctx - selection_reserve(state.rag)
    history = list(state.history_messages)

    while True:
        bases = {
            candidate: count_tokens(
                format_generation_messages(prompt, context="", history=history, question=state.question)
            )
            for candidate, prompt in prompts.items()
        }
        widest = max(bases, key=lambda candidate: bases[candidate])
        remaining = input_budget - bases[widest]

        if remaining > 0:
            return GenerationBudget(
                prompt=prompts[style],
                selection_prompt=prompts[widest],
                history=history,
                num_ctx=num_ctx,
                num_predict=output_ceiling(state.rag, style),
                input_budget=input_budget,
                remaining=remaining,
            )

        if not history:
            raise ContextBudgetExceededError

        history = history[2:]
