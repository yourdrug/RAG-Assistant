"""Shared base prompt and history budget for retrieval and generation."""

from collections.abc import Callable

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from domain.rag.breadth import AnswerStyle, answer_style
from domain.rag.exceptions import ContextBudgetExceededError
from infrastructure.ml.rag.evidence_focus import format_generation_messages
from infrastructure.ml.rag.prompt_budget import estimate_message_tokens
from infrastructure.ml.rag.rag_prompts import build_prompt, domain_prompt_addendum
from infrastructure.ml.rag.retrieval_work import RetrievalWork


def prepare_generation_budget(
    state: RetrievalWork,
    count_tokens: Callable[[list[BaseMessage]], int] = estimate_message_tokens,
) -> tuple[ChatPromptTemplate, list[BaseMessage], int, int, int]:
    style = answer_style(state.ctx.depth)
    registry = state.domain_registry
    keys = state.domain_keys
    prompt = build_prompt(
        style,
        summary=state.ctx.summary,
        domain_addendum=domain_prompt_addendum(state.query_domain, state.ctx, keys, registry),
        enumerate_cases=state.enumerate_cases,
        question=state.question,
    )
    # One context window for every style: the answer style must never decide which evidence
    # fits. The style only reserves its own output ceiling.
    num_ctx = state.rag.llm_num_ctx_broad
    num_predict = (
        state.rag.llm_num_predict_narrow if style == AnswerStyle.CONCISE else state.rag.llm_num_predict_broad
    )
    history = list(state.history_messages)

    while True:
        base = count_tokens(
            format_generation_messages(prompt, context="", history=history, question=state.question)
        )
        remaining = num_ctx - num_predict - base

        if remaining > 0:
            return prompt, history, num_ctx, num_predict, remaining

        if not history:
            raise ContextBudgetExceededError

        history = history[2:]
