"""Translate existing document fixtures into typed stage inputs."""

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from application.rag.dto import (
    ContextBudget,
    PreparedContext,
    PreparedQuery,
    RagRequest,
    RetrievalResult,
    TerminalResponse,
)
from domain.rag.breadth import Breadth
from domain.rag.stream_events import StreamEvent, TextChunk
from infrastructure.ml.rag.rag_steps import (
    RagOperations,
    lc_documents,
    lc_messages,
    message_values,
    scored_values,
)


def stage_fixture(**values: Any) -> SimpleNamespace:
    defaults = {
        "history_messages": [],
        "docs": [],
        "sources": [],
        "sub_queries": [],
        "candidates": [],
        "query_for_search": "",
        "query_domain": "",
        "breadth": Breadth.NARROW,
        "enumerate_cases": False,
        "fetch_k": 0,
        "rerank_top_n": 0,
        "effective_dense_weight": 0.0,
        "effective_sparse_weight": 0.0,
        "use_exact_ref_boost": False,
        "avg_sim": 0.0,
        "req_id": "",
        "q_hash": "",
        "vis_hash": "",
        "cache_revision": None,
        "full_answer": "",
        "grounded_answer": None,
        "evidence_assessment": None,
        "usage_report": None,
        "confidence": 0.0,
        "terminal": False,
        "_prompt_docs": [],
    }
    return SimpleNamespace(**(defaults | values))


def prepared_query(state: SimpleNamespace) -> PreparedQuery:
    request = RagRequest(
        state.question,
        message_values(state.history_messages),
        state.ctx,
        state.rag,
        getattr(state, "t_pipeline_start", 0),
        getattr(state, "req_id", ""),
        "test-model",
    )
    return PreparedQuery(
        request,
        state.query_for_search,
        getattr(state, "q_hash", ""),
        getattr(state, "vis_hash", ""),
        getattr(state, "cache_revision", None),
    )


def retrieval_result(state: SimpleNamespace) -> RetrievalResult:
    return RetrievalResult(
        prepared_query(state),
        state.breadth or Breadth.NARROW,
        state.query_domain,
        state.fetch_k,
        state.rerank_top_n,
        state.effective_dense_weight,
        state.effective_sparse_weight,
        state.use_exact_ref_boost,
        (),
        scored_values(state.docs),
        tuple(state.sub_queries),
        state.enumerate_cases,
        state.avg_sim,
    )


def build_context(
    state: SimpleNamespace, domain_keys: tuple[str, ...] | None, token_counter: Any = None
) -> tuple[list[Any], list[Any]]:
    result = RagOperations(None, domain_keys=domain_keys, document_access=MagicMock()).build(
        retrieval_result(state), token_counter
    )
    state.context = result
    state._prompt_docs = lc_documents(result.documents)
    state.history_messages = lc_messages(result.history)
    state.grounded_answer = result.grounded_answer
    state.evidence_assessment = result.evidence
    return lc_messages(result.messages), state.docs


async def check_cache(
    state: SimpleNamespace, domain_keys: tuple[str, ...] | None = None, document_access: Any = None
) -> tuple[SimpleNamespace, Any]:
    outcome = await RagOperations(None, domain_keys=domain_keys, document_access=document_access).lookup(
        prepared_query(state)
    )

    if isinstance(outcome, TerminalResponse):
        state.terminal = True
        return state, outcome.events

    state.q_hash, state.vis_hash, state.cache_revision = (
        outcome.q_hash,
        outcome.vis_hash,
        outcome.cache_revision,
    )
    return state, None


async def reject_ood(state: SimpleNamespace) -> tuple[SimpleNamespace, TerminalResponse | None]:
    outcome = await RagOperations(None, document_access=MagicMock()).reject_ood(retrieval_result(state))
    return state, outcome


async def generate(
    state: SimpleNamespace, ml: Any, messages: list[Any], grouped: list[Any]
) -> AsyncIterator[StreamEvent]:
    context = getattr(
        state,
        "context",
        PreparedContext(
            retrieval_result(state),
            message_values(messages),
            (),
            (),
            ContextBudget(0, 0, 0, 0),
            state.grounded_answer,
            state.evidence_assessment,
        ),
    )
    answer = await RagOperations(ml, document_access=MagicMock()).generate(context)
    state.full_answer = answer.text
    state.output_sanitized = True

    if answer.text:
        yield TextChunk(text=answer.text)
