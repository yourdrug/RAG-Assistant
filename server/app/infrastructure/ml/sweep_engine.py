"""SweepEngine — multi-strategy parameter sweep for RAG benchmarking.

Generalizes the grid-search logic from CLI into a reusable service supporting:
- grid: cartesian product (for ≤4 parameters)
- random: N random points (for 5+ parameters)
- successive_halving: evaluate all on subset, keep top 50%, repeat

Phase A (cheap): in-memory retrieval-only scoring using cached candidates.
Phase B (expensive): full LLM-judge evaluation on top-N configs.
"""

from __future__ import annotations

import itertools
import logging
import random
from collections.abc import Callable

from config import settings
from domain.entities.benchmark_sweep import BenchmarkSweep
from domain.value_objects.benchmark_strategy import BenchmarkStrategy
from langchain.schema import Document as LCDocument

from infrastructure.clients import get_bm25_index, get_embeddings, get_qdrant_client
from infrastructure.ml.benchmark import load_questions
from infrastructure.ml.hybrid import content_hash, rrf_merge

logger = logging.getLogger("default")

# Parameters that can be sweeped in Phase A (retrieval-time, no reindexing)
CHEAP_PARAMS = frozenset(
    {
        "top_k",
        "fetch_k",
        "rrf_k",
        "dense_weight",
        "sparse_weight",
        "rerank_min_score",
        "rerank_score_gap_ratio",
    }
)

# Parameters that require reindexing (expensive, skip Phase A)
EXPENSIVE_PARAMS = frozenset(
    {
        "chunk_size",
        "chunk_overlap",
        "embed_model",
    }
)


def _estimate_grid_size(search_space: dict) -> int:
    """Estimate total combinations for grid strategy."""
    total = 1
    for param, spec in search_space.items():
        if "values" in spec:
            total *= len(spec["values"])
        elif "min" in spec and "max" in spec and "step" in spec:
            total *= max(1, int((spec["max"] - spec["min"]) / spec["step"]) + 1)
    return total


def generate_grid_points(search_space: dict) -> list[dict]:
    """Generate all parameter combinations (cartesian product)."""
    param_lists = {}
    for param, spec in search_space.items():
        if "values" in spec:
            param_lists[param] = spec["values"]
        elif "min" in spec and "max" in spec and "step" in spec:
            param_lists[param] = []
            v = spec["min"]
            while v <= spec["max"] + 1e-9:
                param_lists[param].append(round(v, 6))
                v += spec["step"]
        else:
            param_lists[param] = [spec.get("default", 0)]

    keys = list(param_lists.keys())
    return [dict(zip(keys, combo)) for combo in itertools.product(*param_lists.values())]


def generate_random_points(search_space: dict, n: int) -> list[dict]:
    """Generate N random parameter combinations."""
    points = []
    for _ in range(n):
        point = {}
        for param, spec in search_space.items():
            if "values" in spec:
                point[param] = random.choice(spec["values"])
            elif "min" in spec and "max" in spec:
                if isinstance(spec.get("step"), float) or isinstance(spec.get("min"), float):
                    point[param] = round(random.uniform(spec["min"], spec["max"]), 6)
                else:
                    point[param] = random.randint(int(spec["min"]), int(spec["max"]))
            else:
                point[param] = spec.get("default", 0)
        points.append(point)
    return points


def compute_composite_score(
    metrics: dict,
    weights: dict,
) -> float:
    """Compute weighted composite score from retrieval + generator metrics."""
    score = 0.0
    total_weight = 0.0

    weight_map = {
        "hit_rate": weights.get("hit_rate", 0.0),
        "mrr": weights.get("mrr", 0.0),
        "faithfulness": weights.get("faithfulness", 0.0),
        "relevancy": weights.get("relevancy", 0.0),
        "correctness": weights.get("correctness", 0.0),
        "avg_similarity": weights.get("avg_similarity", 0.0),
    }

    for metric, w in weight_map.items():
        if w <= 0:
            continue
        val = metrics.get(metric)
        if val is None:
            continue
        # Normalize LLM metrics from 0-10 to 0-1
        if metric in ("faithfulness", "relevancy", "correctness"):
            val = val / 10.0
        score += w * val
        total_weight += w

    return round(score / total_weight, 4) if total_weight > 0 else 0.0


class SweepEngine:
    """Multi-strategy parameter sweep engine."""

    def __init__(
        self,
        uow_factory,
        benchmark_service=None,
        config_service=None,
    ):
        self._uow_factory = uow_factory
        self._benchmark_service = benchmark_service
        self._config_service = config_service

    async def run_sweep(
        self,
        sweep: BenchmarkSweep,
        questions_path: str | None = None,
        judge_model: str | None = None,
        progress_callback: Callable[[int, int, dict | None], None] | None = None,
    ) -> list[dict]:
        """Execute a parameter sweep.

        Args:
            sweep: The sweep configuration entity.
            questions_path: Path to questions JSON (fallback if DB empty).
            judge_model: Model for LLM judge.
            progress_callback: Called with (evaluated, total, latest_result) after each config.

        Returns:
            List of result dicts sorted by composite score (best first).
        """
        search_space = sweep.search_space
        strategy = sweep.strategy
        weights = sweep.objective_weights
        dataset = sweep.dataset
        top_n_llm = sweep.top_n_llm

        # Load questions
        questions_data = await self._load_questions(dataset, questions_path)
        if not questions_data:
            logger.error("No questions found for dataset '%s'", dataset)
            return []

        # Filter to questions with source_hint for retrieval metrics
        eval_questions = [q for q in questions_data if q.get("source_hint") is not None]

        # Generate parameter points based on strategy
        if strategy == BenchmarkStrategy.GRID.value:
            all_points = generate_grid_points(search_space)
        elif strategy == BenchmarkStrategy.RANDOM.value:
            n_random = search_space.get("_n_random", 50)
            all_points = generate_random_points(search_space, n_random)
        elif strategy == BenchmarkStrategy.SUCCESSIVE_HALVING.value:
            all_points = generate_grid_points(search_space)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        total_configs = len(all_points)

        # Filter out expensive params that can't be tested in Phase A
        cheap_points = []
        expensive_in_search = set(search_space.keys()) & EXPENSIVE_PARAMS
        if expensive_in_search:
            logger.warning(
                "Expensive params in search space (skip Phase A for these): %s",
                expensive_in_search,
            )

        for point in all_points:
            cheap_points.append({k: v for k, v in point.items() if k in CHEAP_PARAMS})

        # Phase 1: Cache dense + sparse candidates
        max_fetch_k = max(
            (p.get("fetch_k", settings.retriever_fetch_k) for p in cheap_points),
            default=settings.retriever_fetch_k,
        )
        logger.info(
            "Sweep Phase 1: Caching candidates (fetch_k=%d) for %d questions...",
            max_fetch_k,
            len(eval_questions),
        )
        dense_cache, sparse_cache, all_candidates = await self._cache_candidates(eval_questions, max_fetch_k)

        # Phase A: Cheap retrieval-only scoring
        logger.info("Sweep Phase A: Retrieval-scoring %d configs...", total_configs)
        results = []

        for idx, point in enumerate(all_points, 1):
            result = self._score_config_cheap(
                point, eval_questions, dense_cache, sparse_cache, all_candidates, weights
            )
            result["config"] = point
            results.append(result)

            if progress_callback:
                progress_callback(idx, total_configs, result)

            if idx % 50 == 0:
                logger.info("  %d/%d configs evaluated", idx, total_configs)

        # Sort by composite score
        results.sort(key=lambda x: x.get("composite_score", 0), reverse=True)

        logger.info("Phase A done. Top-5 retrieval scores:")
        for i, r in enumerate(results[:5], 1):
            logger.info(
                "  #%d  config=%s  HR=%.3f  MRR=%.4f  composite=%.4f",
                i,
                r["config"],
                r.get("avg_hit_rate", 0),
                r.get("avg_mrr", 0),
                r.get("composite_score", 0),
            )

        # Phase B: Full LLM-judge on top-N (if judge_model provided)
        if judge_model and top_n_llm > 0:
            logger.info("Sweep Phase B: LLM-judge on top-%d configs...", top_n_llm)
            top_configs = results[:top_n_llm]

            for cfg_idx, cfg in enumerate(top_configs, 1):
                logger.info(
                    "  LLM evaluation #%d/%d: %s",
                    cfg_idx,
                    top_n_llm,
                    cfg["config"],
                )

                # Apply config to settings temporarily
                orig = self._snapshot_settings()
                try:
                    self._apply_config(cfg["config"])
                    full_result = self._run_full_benchmark(
                        questions_path or str(settings.data_dir / "test_questions.json"),
                        judge_model,
                    )
                    cfg["full_metrics"] = full_result
                    cfg["llm_evaluated"] = True
                    # Update composite with LLM scores
                    cfg["composite_score"] = compute_composite_score(
                        {
                            "hit_rate": cfg.get("avg_hit_rate", 0),
                            "mrr": cfg.get("avg_mrr", 0),
                            "faithfulness": full_result.get("avg_faithfulness", 0),
                            "relevancy": full_result.get("avg_relevancy", 0),
                            "correctness": full_result.get("avg_correctness"),
                        },
                        weights,
                    )
                finally:
                    self._restore_settings(orig)

            # Re-sort with LLM scores
            results.sort(key=lambda x: x.get("composite_score", 0), reverse=True)

        return results

    async def _load_questions(self, dataset: str, questions_path: str | None) -> list[dict]:
        """Load questions from DB, falling back to file."""
        try:
            async with self._uow_factory.create() as uow:
                questions = await uow.benchmark_questions.list(dataset=dataset, is_active=True, limit=1000)
            if questions:
                return [
                    {
                        "question": q.question,
                        "expected_answer": q.expected_answer,
                        "source_hint": q.source_hint,
                        "tags": q.tags or [],
                    }
                    for q in questions
                ]
        except Exception as e:
            logger.warning("Failed to load questions from DB: %s", e)

        # Fallback to file
        path = questions_path or str(settings.data_dir / "test_questions.json")
        return load_questions(path)

    async def _cache_candidates(self, questions: list[dict], max_fetch_k: int) -> tuple[dict, dict, dict]:
        """Phase 1: Cache dense + sparse candidates at max fetch_k."""
        dense_cache: dict[str, list] = {}
        sparse_cache: dict[str, list] = {}
        all_candidates: dict[str, LCDocument] = {}

        client = get_qdrant_client()
        embeddings = get_embeddings()

        for q in questions:
            qtext = q["question"]

            # Dense search
            dense_results = []
            for point in client.search(
                collection_name=settings.collection_name,
                query_vector=embeddings.embed_query(qtext),
                limit=max_fetch_k,
            ):
                payload = point.payload or {}
                page_content = payload.get("page_content", "")
                metadata = payload.get("metadata", {})
                h = metadata.get("content_hash") or content_hash(page_content)
                doc = LCDocument(page_content=page_content, metadata=metadata)
                dense_results.append((h, point.score, doc))
                all_candidates[h] = doc
            dense_cache[qtext] = dense_results

            # Sparse search
            bm25 = get_bm25_index()
            if bm25:
                sparse_results = bm25.search_with_hashes(qtext, max_fetch_k)
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

    def _score_config_cheap(
        self,
        config: dict,
        questions: list[dict],
        dense_cache: dict,
        sparse_cache: dict,
        all_candidates: dict,
        weights: dict,
    ) -> dict:
        """Phase A: Score a config using cached candidates (no LLM/Qdrant calls)."""
        top_k = config.get("top_k", settings.retriever_top_k)
        fetch_k = config.get("fetch_k", settings.retriever_fetch_k)
        dw = config.get("dense_weight", settings.dense_weight)
        sw = config.get("sparse_weight", settings.sparse_weight)
        rrf_k = config.get("rrf_k", settings.rrf_k)

        hit_rates = []
        mrrs = []

        for q in questions:
            qtext = q["question"]
            source_hint = q.get("source_hint")
            if source_hint is None:
                continue

            # Trim to fetch_k
            dense_trimmed = dense_cache.get(qtext, [])[:fetch_k]
            sparse_trimmed = sparse_cache.get(qtext, [])[:fetch_k]

            # RRF merge
            merged_hashes = rrf_merge(
                [(h, score) for h, score, _doc in dense_trimmed],
                sparse_trimmed,
                k=rrf_k,
                dense_weight=dw,
                sparse_weight=sw,
            )

            # Dedup and take top_k
            seen = set()
            top_hashes = []
            for h in merged_hashes:
                if h not in seen:
                    seen.add(h)
                    top_hashes.append(h)
                    if len(top_hashes) >= top_k:
                        break

            # Compute hit_rate and MRR
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

    def _snapshot_settings(self) -> dict:
        """Snapshot current settings for later restoration."""
        return {
            "top_k": settings.retriever_top_k,
            "fetch_k": settings.retriever_fetch_k,
            "dense": settings.dense_weight,
            "sparse": settings.sparse_weight,
            "rrf": settings.rrf_k,
            "min_score": settings.rerank_min_score,
            "gap_ratio": settings.rerank_score_gap_ratio,
        }

    def _apply_config(self, config: dict) -> None:
        """Temporarily apply a config to global settings."""
        if "top_k" in config:
            settings.retriever_top_k = config["top_k"]
        if "fetch_k" in config:
            settings.retriever_fetch_k = config["fetch_k"]
        if "dense_weight" in config:
            settings.dense_weight = config["dense_weight"]
        if "sparse_weight" in config:
            settings.sparse_weight = config["sparse_weight"]
        if "rrf_k" in config:
            settings.rrf_k = config["rrf_k"]
        if "rerank_min_score" in config:
            settings.rerank_min_score = config["rerank_min_score"]
        if "rerank_score_gap_ratio" in config:
            settings.rerank_score_gap_ratio = config["rerank_score_gap_ratio"]

    def _restore_settings(self, snapshot: dict) -> None:
        """Restore settings from snapshot."""
        settings.retriever_top_k = snapshot["top_k"]
        settings.retriever_fetch_k = snapshot["fetch_k"]
        settings.dense_weight = snapshot["dense"]
        settings.sparse_weight = snapshot["sparse"]
        settings.rrf_k = snapshot["rrf"]
        settings.rerank_min_score = snapshot["min_score"]
        settings.rerank_score_gap_ratio = snapshot["gap_ratio"]

    def _run_full_benchmark(self, questions_path: str, judge_model: str) -> dict:
        """Run a full benchmark with LLM judge (blocking)."""
        if self._benchmark_service is None:
            from infrastructure.services.benchmark_service import BenchmarkService

            self._benchmark_service = BenchmarkService()

        out_dir = str(settings.data_dir / "benchmark_results")
        result = self._benchmark_service.run(
            questions_path=questions_path,
            out_dir=out_dir,
            top_k=settings.retriever_top_k,
            judge_model=judge_model,
        )
        return result
