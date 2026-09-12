"""Sweep scoring — grid generation, composite scoring, and Phase A caching.

Extracted from ``sweep_engine.py`` to reduce its size. These are internal
helpers used by the sweep engine; they are NOT part of the public API.
"""

from __future__ import annotations

import itertools
import logging
from typing import TYPE_CHECKING

from config import settings
from infrastructure.bm25.hybrid import content_hash, rrf_merge

if TYPE_CHECKING:
    from langchain.schema import Document as LCDocument

logger = logging.getLogger("default")


def generate_grid_points(search_space: dict) -> list[dict]:
    """Generate all parameter combinations (cartesian product)."""
    param_lists = {}
    for param, spec in search_space.items():
        if param.startswith("_"):
            continue
        if "values" in spec:
            param_lists[param] = spec["values"]
        elif "min" in spec and "max" in spec:
            step = spec.get("step", 1)
            param_lists[param] = list(range(int(spec["min"]), int(spec["max"]) + 1, int(step)))
        else:
            param_lists[param] = [spec.get("default", 0)]

    if not param_lists:
        return [{}]

    keys = list(param_lists.keys())
    value_combos = list(itertools.product(*[param_lists[k] for k in keys]))
    return [dict(zip(keys, combo, strict=False)) for combo in value_combos]


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
    """Compute weighted composite score from individual metrics."""
    if weights is None:
        weights = {}

    default_weights = {
        "hit_rate": 0.3,
        "mrr": 0.2,
        "faithfulness": 0.25,
        "relevancy": 0.15,
        "correctness": 0.1,
    }
    w = {**default_weights, **(weights or {})}

    total = 0.0
    total_weight = 0.0
    for key, weight in w.items():
        value = metrics.get(key)
        if value is not None:
            total += float(value) * weight
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
) -> dict:
    """Phase A: Score a config using cached candidates (no LLM/Qdrant calls).

    *defaults* provides fallback values for params not in *config*.
    Falls back to global settings if not provided.
    """
    if defaults is None:
        defaults = {}
    top_k = config.get("top_k", defaults.get("retriever_top_k", settings.retriever_top_k))
    fetch_k = config.get("fetch_k", defaults.get("retriever_fetch_k", settings.retriever_fetch_k))
    dw = config.get("dense_weight", defaults.get("dense_weight", settings.dense_weight))
    sw = config.get("sparse_weight", defaults.get("sparse_weight", settings.sparse_weight))
    rrf_k = config.get("rrf_k", defaults.get("rrf_k", settings.rrf_k))

    hit_rates = []
    mrrs = []

    for q in questions:
        qtext = q["question"]
        source_hint = q.get("source_hint")
        if source_hint is None:
            continue

        dense_trimmed = dense_cache.get(qtext, [])[:fetch_k]
        sparse_trimmed = sparse_cache.get(qtext, [])[:fetch_k]

        merged_hashes = rrf_merge(
            [(h, score) for h, score, _doc in dense_trimmed],
            sparse_trimmed,
            k=rrf_k,
            dense_weight=dw,
            sparse_weight=sw,
        )

        seen = set()
        top_hashes = []
        for h in merged_hashes:
            if h not in seen:
                seen.add(h)
                top_hashes.append(h)
                if len(top_hashes) >= top_k:
                    break

        hit = 0
        mrr = 0.0
        for rank, h in enumerate(top_hashes, 1):
            doc = all_candidates.get(h)
            if doc is None:
                continue
            filename = doc.metadata.get("filename", "") or doc.metadata.get("source", "")
            if source_hint.lower() in filename.lower():
                hit = 1
                if mrr == 0.0:
                    mrr = 1.0 / rank
                break

        hit_rates.append(hit)
        mrrs.append(mrr)

    avg_hr = sum(hit_rates) / len(hit_rates) if hit_rates else 0
    avg_mrr = sum(mrrs) / len(mrrs) if mrrs else 0

    metrics = {
        "avg_hit_rate": round(avg_hr, 3),
        "avg_mrr": round(avg_mrr, 4),
    }
    metrics["composite_score"] = compute_composite_score({"hit_rate": avg_hr, "mrr": avg_mrr}, weights)
    return metrics


async def cache_candidates(
    questions: list[dict],
    max_fetch_k: int,
    ml_clients=None,
) -> tuple[dict, dict, dict]:
    """Phase 1: Cache dense + sparse candidates at max fetch_k."""
    dense_cache: dict[str, list] = {}
    sparse_cache: dict[str, list] = {}
    all_candidates: dict[str, "LCDocument"] = {}

    if ml_clients is not None:
        client = ml_clients.qdrant_client()
        embeddings = ml_clients.embeddings()
        bm25_index = ml_clients.bm25_index()
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
        for point in client.search(
            collection_name=settings.collection_name,
            query_vector=embeddings.embed_query_sync(qtext),
            limit=max_fetch_k,
        ):
            payload = point.payload or {}
            page_content = payload.get("page_content", "")
            metadata = payload.get("metadata", {})
            h = metadata.get("content_hash") or content_hash(page_content)
            from langchain.schema import Document as LCDocument

            doc = LCDocument(page_content=page_content, metadata=metadata)
            dense_results.append((h, point.score, doc))
            all_candidates[h] = doc
        dense_cache[qtext] = dense_results

        if bm25_index:
            sparse_results = bm25_index.search_with_hashes(qtext, max_fetch_k)
        else:
            sparse_results = []
        sparse_cache[qtext] = sparse_results

    logger.info(
        "Cache built: %d dense, %d sparse, %d unique hashes",
        sum(len(v) for v in dense_cache.values()),
        sum(len(v) for v in sparse_cache.values()),
        len(all_candidates),
    )
    return dense_cache, sparse_cache, all_candidates
