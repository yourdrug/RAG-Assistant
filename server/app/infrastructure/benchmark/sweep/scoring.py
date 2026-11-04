"""Pure scoring of cached retrieval candidates and benchmark objectives."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NotRequired, TypedDict, Unpack

from langchain_core.documents import Document as LCDocument

from application.search import HybridRetriever
from domain.benchmark.evaluation import (
    evaluate_retrieval,
    has_retrieval_labels,
    summarize_evidence,
    summarize_retrieval,
)
from domain.benchmark.scoring import (
    METRIC_SCALES,
    compute_composite_score,
)
from domain.rag.ranking import rrf_merge
from infrastructure.ml.rag.rag_reranking import filter_scope_candidates


if TYPE_CHECKING:
    from application.benchmark.sweep_dto import CandidateCache, DenseCache, RerankerCache, SparseCache
    from infrastructure.benchmark.sweep.candidates import DenseDocumentCache, DocumentCache


class ScoringOptions(TypedDict):
    weights: dict[str, Any]
    defaults: dict[str, Any]
    rerank_scores: NotRequired[RerankerCache | None]


def score_retrieval_evidence(
    question: dict[str, Any], documents: list[dict[str, Any]]
) -> tuple[int | None, float | None]:
    """Compatibility entry point for the shared domain evaluator."""
    metrics = evaluate_retrieval(question, documents)
    return metrics["hit_rate"], metrics["mrr"]


def score_config_cheap(
    config: dict[str, Any],
    questions: list[dict[str, Any]],
    dense_cache: DenseDocumentCache,
    sparse_cache: SparseCache,
    all_candidates: DocumentCache,
    **options: Unpack[ScoringOptions],
) -> dict[str, Any]:
    """Phase A: Score a config using cached candidates (no LLM/Qdrant calls).

    *defaults* provides fallback values for params not in *config*.
    Cached reranker
    scores enable the production merge, top-k and threshold filtering path.
    """
    weights = options["weights"]
    defaults = options["defaults"]
    rerank_scores = options.get("rerank_scores")

    if rerank_scores is None and {"rerank_min_score", "rerank_score_gap_ratio"} & config.keys():
        message = "Cached reranker scores are required to compare reranker thresholds"
        raise ValueError(message)

    parameters = RetrievalParameters.from_config(config, defaults)

    case_results = []

    for q in questions:
        if not has_retrieval_labels(q):
            continue

        documents = retrieval_documents(
            q["question"], dense_cache, sparse_cache, all_candidates, parameters, rerank_scores=rerank_scores
        )

        retrieval = evaluate_retrieval(q, documents)
        case_results.append(
            {
                "annotations": q.get("annotations"),
                "retriever_metrics": retrieval,
                "evidence_metrics": retrieval,
            }
        )

    summary = {**summarize_retrieval(case_results), **summarize_evidence(case_results)}
    metrics = {"avg_hit_rate": summary["hit_rate"], **summary}
    metrics["composite_score"] = compute_composite_score(
        {key: summary.get("hit_rate" if key == "hit_rate" else f"avg_{key}") for key in METRIC_SCALES},
        weights,
    )
    return metrics


def score_stage_candidates(
    config: dict[str, Any],
    questions: list[dict[str, Any]],
    dense_cache: DenseCache,
    sparse_cache: SparseCache,
    all_candidates: CandidateCache,
    **options: Unpack[ScoringOptions],
) -> dict[str, Any]:
    documents = {
        key: LCDocument(page_content=chunk.content, metadata=deepcopy(chunk.metadata))
        for key, chunk in all_candidates.items()
    }
    dense = {
        query: [(key, score, documents[key]) for key, score, _ in hits] for query, hits in dense_cache.items()
    }
    return score_config_cheap(config, questions, dense, sparse_cache, documents, **options)


__all__ = ["compute_composite_score", "has_retrieval_labels"]


@dataclass(frozen=True)
class RetrievalParameters:
    top_k: int
    fetch_k: int
    dw: float
    sw: float
    rrf_k: int
    min_score: float
    gap_ratio: float

    @classmethod
    def from_config(cls, config: dict[str, Any], defaults: dict[str, Any]) -> RetrievalParameters:
        return cls(
            top_k=config.get("top_k", defaults["retriever_top_k"]),
            fetch_k=config.get("fetch_k", defaults["retriever_fetch_k"]),
            dw=config.get("dense_weight", defaults["dense_weight"]),
            sw=config.get("sparse_weight", defaults["sparse_weight"]),
            rrf_k=config.get("rrf_k", defaults["rrf_k"]),
            min_score=config.get("rerank_min_score", defaults["rerank_min_score"]),
            gap_ratio=config.get("rerank_score_gap_ratio", defaults["rerank_score_gap_ratio"]),
        )


def retrieval_documents(
    qtext: str,
    dense_cache: DenseDocumentCache,
    sparse_cache: SparseCache,
    all_candidates: DocumentCache,
    parameters: RetrievalParameters,
    *,
    rerank_scores: RerankerCache | None,
) -> list[dict[str, Any]]:
    retriever = HybridRetriever()
    dense_trimmed = dense_cache.get(qtext, [])[: parameters.fetch_k]
    sparse_trimmed = sparse_cache.get(qtext, [])[: parameters.fetch_k]

    if rerank_scores is None:
        merged_hashes = rrf_merge(
            [(h, score) for h, score, _doc in dense_trimmed],
            sparse_trimmed,
            k=parameters.rrf_k,
            dense_weight=parameters.dw,
            sparse_weight=parameters.sw,
        )
        top_hashes = list(dict.fromkeys(merged_hashes))[: parameters.top_k]
    else:
        available = {
            h: (0.0, all_candidates[h])
            for h in dict.fromkeys([h for h, _, _ in dense_trimmed] + [h for h, _ in sparse_trimmed])
            if h in all_candidates
        }
        candidates = retriever.merge_and_dedup(
            [(h, score) for h, score, _ in dense_trimmed],
            sparse_trimmed,
            available,
            fetch_k=parameters.fetch_k,
            rrf_k=parameters.rrf_k,
            dense_weight=parameters.dw,
            sparse_weight=parameters.sw,
        )
        # The reranker cache excludes contradictory scope before inference.
        # Mirror production's post-merge scope filter before reading scores.
        candidates = filter_scope_candidates(candidates, qtext)
        hash_by_document = {id(doc): h for h, (_, doc) in available.items()}
        query_scores = rerank_scores.get(qtext, {})
        ranked = sorted(
            [(hash_by_document[id(doc)], query_scores[hash_by_document[id(doc)]]) for doc in candidates],
            key=lambda item: item[1],
            reverse=True,
        )[: parameters.top_k]
        top_hashes = [
            h for h, _ in retriever.apply_rerank_filters(ranked, parameters.min_score, parameters.gap_ratio)
        ]

    return project_candidates(top_hashes, all_candidates)


def project_candidates(top_hashes: list[str], all_candidates: DocumentCache) -> list[dict[str, Any]]:
    documents = []

    for h in top_hashes:
        doc = all_candidates.get(h)

        if doc is None:
            # Preserve the original rank even for unresolved sparse hits.
            documents.append({"content": "", "metadata": {"source": ""}})
            continue

        metadata = dict(doc.metadata)
        metadata["source"] = metadata.get("filename") or metadata.get("source", "")
        documents.append({"content": doc.page_content, "metadata": metadata})

    return documents
