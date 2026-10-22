"""Pure scoring of cached retrieval candidates and benchmark objectives."""

from __future__ import annotations
from copy import deepcopy
from langchain_core.documents import Document as LCDocument


from application.search import HybridRetriever
from domain.benchmark.evaluation import (
    evaluate_retrieval,
    has_retrieval_labels as has_retrieval_labels,
    summarize_evidence,
    summarize_retrieval,
)
from domain.rag.retrieval import rrf_merge
from domain.benchmark.scoring import (
    compute_composite_score as compute_composite_score,
    METRIC_SCALES,
)
from infrastructure.ml.rag.rag_reranking import filter_scope_candidates


def score_retrieval_evidence(question: dict, documents: list[dict]) -> tuple[int | None, float | None]:
    """Compatibility entry point for the shared domain evaluator."""
    metrics = evaluate_retrieval(question, documents)
    return metrics["hit_rate"], metrics["mrr"]


def score_config_cheap(  # noqa: C901
    config: dict,
    questions: list[dict],
    dense_cache: dict,
    sparse_cache: dict,
    all_candidates: dict,
    weights: dict,
    defaults: dict,
    *,
    rerank_scores: dict | None = None,
) -> dict:
    """Phase A: Score a config using cached candidates (no LLM/Qdrant calls).

    *defaults* provides fallback values for params not in *config*.
    Cached reranker
    scores enable the production merge, top-k and threshold filtering path.
    """
    if rerank_scores is None and {"rerank_min_score", "rerank_score_gap_ratio"} & config.keys():
        raise ValueError("Cached reranker scores are required to compare reranker thresholds")

    top_k = config.get("top_k", defaults["retriever_top_k"])
    fetch_k = config.get("fetch_k", defaults["retriever_fetch_k"])
    dw = config.get("dense_weight", defaults["dense_weight"])
    sw = config.get("sparse_weight", defaults["sparse_weight"])
    rrf_k = config.get("rrf_k", defaults["rrf_k"])
    min_score = config.get("rerank_min_score", defaults["rerank_min_score"])
    gap_ratio = config.get(
        "rerank_score_gap_ratio",
        defaults["rerank_score_gap_ratio"],
    )
    retriever = HybridRetriever()

    case_results = []

    for q in questions:
        qtext = q["question"]

        if not has_retrieval_labels(q):
            continue

        dense_trimmed = dense_cache.get(qtext, [])[:fetch_k]
        sparse_trimmed = sparse_cache.get(qtext, [])[:fetch_k]

        if rerank_scores is None:
            merged_hashes = rrf_merge(
                [(h, score) for h, score, _doc in dense_trimmed],
                sparse_trimmed,
                k=rrf_k,
                dense_weight=dw,
                sparse_weight=sw,
            )
            top_hashes = list(dict.fromkeys(merged_hashes))[:top_k]
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
                fetch_k=fetch_k,
                rrf_k=rrf_k,
                dense_weight=dw,
                sparse_weight=sw,
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
            )[:top_k]
            top_hashes = [h for h, _ in retriever.apply_rerank_filters(ranked, min_score, gap_ratio)]

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
    config, questions, dense_cache, sparse_cache, all_candidates, weights, *, rerank_scores, defaults
):
    documents = {
        key: LCDocument(page_content=chunk.content, metadata=deepcopy(chunk.metadata))
        for key, chunk in all_candidates.items()
    }
    dense = {
        query: [(key, score, documents[key]) for key, score, _ in hits] for query, hits in dense_cache.items()
    }
    return score_config_cheap(
        config, questions, dense, sparse_cache, documents, weights, defaults, rerank_scores=rerank_scores
    )
