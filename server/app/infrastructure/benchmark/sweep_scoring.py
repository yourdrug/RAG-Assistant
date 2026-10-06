"""Sweep scoring — grid generation, composite scoring, and Phase A caching.

Extracted from ``sweep_engine.py`` to reduce its size. These are internal
helpers used by the sweep engine; they are NOT part of the public API.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
from decimal import Decimal

from langchain.schema import Document as LCDocument

from application.services.retrieval import HybridRetriever
from config import get_setting, settings
from domain.services.benchmark_evaluation import (
    evaluate_retrieval,
    has_retrieval_labels as has_retrieval_labels,
    summarize_evidence,
    summarize_retrieval,
)
from domain.utils import content_hash, rrf_merge
from domain.value_objects.benchmark_scoring import (
    DEFAULT_OBJECTIVE_WEIGHTS,
    METRIC_SCALES,
    validate_objective_weights,
)


logger = logging.getLogger("default")


def score_retrieval_evidence(question: dict, documents: list[dict]) -> tuple[int | None, float | None]:
    """Compatibility entry point for the shared domain evaluator."""
    metrics = evaluate_retrieval(question, documents)
    return metrics["hit_rate"], metrics["mrr"]


def generate_grid_points(search_space: dict) -> list[dict]:
    """Generate all parameter combinations (cartesian product)."""
    param_lists = {}
    for param, spec in search_space.items():
        if param.startswith("_"):
            continue
        if "values" in spec:
            param_lists[param] = spec["values"]
        elif "min" in spec and "max" in spec:
            param_lists[param] = grid_range_values(spec)
        else:
            param_lists[param] = [spec.get("default", 0)]

    if not param_lists:
        return [{}]

    keys = list(param_lists.keys())
    value_combos = list(itertools.product(*[param_lists[k] for k in keys]))
    return [dict(zip(keys, combo, strict=False)) for combo in value_combos]


def grid_range_values(spec: dict) -> list[int | float]:
    """Enumerate inclusive ranges without truncating floating rerank thresholds."""
    start, stop, step = (Decimal(str(value)) for value in (spec["min"], spec["max"], spec.get("step", 1)))
    if not all(value.is_finite() for value in (start, stop, step)) or step <= 0 or stop < start:
        raise ValueError("Grid ranges require finite bounds, min <= max and a positive step")
    is_float = any(isinstance(spec.get(key), float) for key in ("min", "max", "step"))
    count = int((stop - start) // step) + 1
    convert = float if is_float else int
    return [convert(start + step * index) for index in range(count)]


def generate_random_points(search_space: dict, n: int) -> list[dict]:
    """Generate N random parameter combinations."""
    import random as _random

    points = []
    for _ in range(n):
        point = {}
        for param, spec in search_space.items():
            if param.startswith("_"):
                continue
            if "values" in spec:
                point[param] = _random.choice(spec["values"])  # noqa: S311
            elif "min" in spec and "max" in spec:
                step = spec.get("step")
                is_float = (
                    isinstance(spec.get("min"), float)
                    or isinstance(spec.get("max"), float)
                    or (step and isinstance(step, float))
                )
                if is_float:
                    point[param] = _random.uniform(spec["min"], spec["max"])  # noqa: S311
                else:
                    point[param] = _random.randint(int(spec["min"]), int(spec["max"]))  # noqa: S311
            else:
                point[param] = spec.get("default", 0)
        points.append(point)
    return points


def compute_composite_score(
    metrics: dict,
    weights: dict | None = None,
) -> float:
    """Weighted mean on a common 0–1 scale, using only available metrics.

    Phase A uses retrieval metrics only. Final selection requires all
    positively weighted metrics to be available (see SweepFullEvaluator).
    """
    resolved = validate_objective_weights(
        DEFAULT_OBJECTIVE_WEIGHTS if weights is None else weights, require_retrieval=False
    )
    total = 0.0
    total_weight = 0.0
    for key, weight in resolved.items():
        value = metrics.get(key)
        if weight > 0 and value is not None:
            total += float(value) / METRIC_SCALES[key] * weight
            total_weight += weight
    return total / total_weight if total_weight > 0 else 0.0


def score_config_cheap(  # noqa: C901
    config: dict,
    questions: list[dict],
    dense_cache: dict,
    sparse_cache: dict,
    all_candidates: dict,
    weights: dict,
    defaults: dict | None = None,
    *,
    rerank_scores: dict | None = None,
) -> dict:
    """Phase A: Score a config using cached candidates (no LLM/Qdrant calls).

    *defaults* provides fallback values for params not in *config*.
    Falls back to effective runtime settings if not provided. Cached reranker
    scores enable the production merge, top-k and threshold filtering path.
    """
    if defaults is None:
        defaults = {}
    if rerank_scores is None and {"rerank_min_score", "rerank_score_gap_ratio"} & config.keys():
        raise ValueError("Cached reranker scores are required to compare reranker thresholds")
    top_k = config.get("top_k", defaults.get("retriever_top_k", get_setting("retriever_top_k")))
    fetch_k = config.get("fetch_k", defaults.get("retriever_fetch_k", get_setting("retriever_fetch_k")))
    dw = config.get("dense_weight", defaults.get("dense_weight", get_setting("dense_weight")))
    sw = config.get("sparse_weight", defaults.get("sparse_weight", get_setting("sparse_weight")))
    rrf_k = config.get("rrf_k", defaults.get("rrf_k", get_setting("rrf_k")))
    min_score = config.get(
        "rerank_min_score", defaults.get("rerank_min_score", get_setting("rerank_min_score"))
    )
    gap_ratio = config.get(
        "rerank_score_gap_ratio",
        defaults.get("rerank_score_gap_ratio", get_setting("rerank_score_gap_ratio")),
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
                fetch_k,
                rrf_k,
                dw,
                sw,
            )
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


async def cache_candidates(
    questions: list[dict],
    max_fetch_k: int,
    ml_clients=None,
    access_filter=None,
    visibility_conditions=None,
    user_id: int | None = None,
    user_group_ids: list[int] | None = None,
) -> tuple[dict, dict, dict]:
    """Phase 1: Cache dense + sparse candidates at max fetch_k.

    ACL parameters are forwarded to Qdrant search and BM25 pre-filter
    to ensure sweep retrieval respects document visibility.
    """
    dense_cache: dict[str, list] = {}
    sparse_cache: dict[str, list] = {}
    all_candidates: dict[str, "LCDocument"] = {}

    if ml_clients is not None:
        client = ml_clients.qdrant_client()
        embeddings = ml_clients.embeddings()
        bm25_index = await ml_clients._ensure_bm25_loaded()
    else:
        from infrastructure.ml.clients.factories import (
            create_embeddings,
            create_qdrant_client,
            load_bm25_index,
        )

        client = create_qdrant_client()
        embeddings = create_embeddings()
        bm25_index = load_bm25_index()

    for q in questions:
        qtext = q["question"]

        dense_results = []
        query_vector = await asyncio.to_thread(embeddings.embed_query_sync, qtext)
        points = await asyncio.to_thread(
            client.search,
            collection_name=settings.collection_name,
            query_vector=query_vector,
            limit=max_fetch_k,
            query_filter=access_filter,
        )
        for point in points:
            payload = point.payload or {}
            page_content = payload.get("page_content", "")
            metadata = payload.get("metadata", {})
            h = metadata.get("content_hash") or content_hash(page_content)

            doc = LCDocument(page_content=page_content, metadata=metadata)
            dense_results.append((h, point.score, doc))
            all_candidates[h] = doc
        dense_cache[qtext] = dense_results

        if bm25_index:
            sparse_results = await asyncio.to_thread(
                bm25_index.search_with_hashes,
                qtext,
                max_fetch_k,
                visibility_conditions=visibility_conditions,
                user_id=user_id,
                user_group_ids=user_group_ids or [],
            )
        else:
            sparse_results = []
        sparse_cache[qtext] = sparse_results

        missing = [h for h, _ in sparse_results if h not in all_candidates]
        if missing:
            if access_filter is None:
                raise RuntimeError("Sweep sparse candidate resolution requires an ACL filter")
            if ml_clients is not None:
                from infrastructure.ml.rag.rag_retrieval import resolve_hashes_batch

                all_candidates.update(await resolve_hashes_batch(missing, access_filter, ml_clients))
            else:
                from qdrant_client.models import FieldCondition, Filter, MatchAny

                points, _ = await asyncio.to_thread(
                    client.scroll,
                    collection_name=settings.collection_name,
                    scroll_filter=Filter(
                        must=[
                            access_filter,
                            FieldCondition(key="metadata.content_hash", match=MatchAny(any=missing)),
                        ]
                    ),
                    limit=len(missing),
                    with_payload=True,
                    timeout=settings.qdrant_timeout,
                )
                for point in points:
                    payload = point.payload or {}
                    metadata = payload.get("metadata", {})
                    h = metadata.get("content_hash")
                    if h:
                        all_candidates[h] = LCDocument(
                            page_content=payload.get("page_content", ""), metadata=metadata
                        )

    logger.info(
        "Cache built: %d dense, %d sparse, %d unique hashes",
        sum(len(v) for v in dense_cache.values()),
        sum(len(v) for v in sparse_cache.values()),
        len(all_candidates),
    )
    return dense_cache, sparse_cache, all_candidates
