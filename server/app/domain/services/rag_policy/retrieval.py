"""Pure policies for retrieval limits, hybrid weights and document context budgets."""

from __future__ import annotations

from typing import TypedDict

from domain.value_objects.llm_provider import Breadth
from domain.value_objects.rag_settings import RagSettings

from .classification import has_exact_reference


class RetrievalParams(TypedDict):
    """Candidate limits and weights consumed by the retrieval pipeline."""

    fetch_k: int
    rerank_top_n: int
    effective_dense_weight: float
    effective_sparse_weight: float
    use_exact_ref_boost: bool


def compute_retrieval_params(
    breadth: Breadth,
    rag_settings: RagSettings,
    query: str,
    exact_ref_sparse_boost: float = 1.0,
) -> RetrievalParams:
    """Compute retrieval parameters based on breadth and settings.

    ``rerank_top_n`` always uses the broad candidate limit; final document
    selection is handled separately by ``select_final_top_k``. Only the
    sparse weight is boosted for exact structural references.

    Pure function — no infrastructure dependencies.
    """
    use_exact_ref_boost = has_exact_reference(query)

    fetch_k = (
        rag_settings.retriever.fetch_k_broad if breadth == Breadth.BROAD else rag_settings.retriever.fetch_k
    )
    rerank_top_n = rag_settings.retriever.top_k_broad

    effective_dense_weight = rag_settings.hybrid_search.dense_weight
    effective_sparse_weight = rag_settings.hybrid_search.sparse_weight
    if use_exact_ref_boost:
        effective_sparse_weight = rag_settings.hybrid_search.sparse_weight * exact_ref_sparse_boost

    return {
        "fetch_k": fetch_k,
        "rerank_top_n": rerank_top_n,
        "effective_dense_weight": effective_dense_weight,
        "effective_sparse_weight": effective_sparse_weight,
        "use_exact_ref_boost": use_exact_ref_boost,
    }


def compute_context_budget(
    breadth: Breadth,
    enumerate_cases: bool,
    history_chars: int,
    question_chars: int,
    num_ctx_narrow: int,
    num_ctx_broad: int,
    chars_per_token: int = 4,
    reserved_overhead: int = 3000,
    min_context_tokens: int = 1000,
    num_predict_narrow: int = 400,
    num_predict_broad: int = 2048,
) -> int:
    """Compute maximum context tokens available for retrieved documents.

    Pure function — no infrastructure dependencies.
    """
    effective_breadth = Breadth.BROAD if enumerate_cases else breadth
    num_ctx = num_ctx_broad if effective_breadth == Breadth.BROAD else num_ctx_narrow
    num_predict = num_predict_broad if effective_breadth == Breadth.BROAD else num_predict_narrow
    reserved_chars = history_chars + question_chars + reserved_overhead
    reserved_for_system_and_history = max(reserved_chars // chars_per_token, 1500)
    return max(num_ctx - num_predict - reserved_for_system_and_history, min_context_tokens)


def select_final_top_k(
    breadth: Breadth,
    enumerate_cases: bool,
    rag_settings: RagSettings,
) -> int:
    """Select the final number of documents to keep after reranking.

    Pure function — no infrastructure dependencies.
    """
    if not enumerate_cases:
        return (
            rag_settings.retriever.top_k if breadth == Breadth.NARROW else rag_settings.retriever.top_k_broad
        )
    return rag_settings.retriever.top_k_broad
