"""Pure policies for retrieval limits and hybrid weights."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from domain.rag.breadth import Breadth
from domain.rag.classification import has_exact_reference


if TYPE_CHECKING:
    from domain.rag.settings import RagSettings


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
) -> RetrievalParams:
    """Compute retrieval parameters based on breadth and settings.

    ``rerank_top_n`` always uses the broad candidate limit; the reranker's
    score thresholds and the generation token budget bound the final set.
    Only the sparse weight is boosted for exact structural references.

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
        effective_sparse_weight = (
            rag_settings.hybrid_search.sparse_weight * rag_settings.exact_ref_sparse_boost
        )

    return {
        "fetch_k": fetch_k,
        "rerank_top_n": rerank_top_n,
        "effective_dense_weight": effective_dense_weight,
        "effective_sparse_weight": effective_sparse_weight,
        "use_exact_ref_boost": use_exact_ref_boost,
    }
