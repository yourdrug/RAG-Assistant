"""Deterministic shortlist selection with relevance and parameter diversity."""

import json
import math

FAST_ELITE_FRACTION = 0.5
FAST_COMPETITIVE_SCORE_RATIO = 0.9


def retrieval_priority(result: dict) -> tuple[float, ...]:
    return tuple(
        float(result.get(key) or 0)
        for key in (
            "composite_score",
            "avg_fragment_recall_at_k",
            "avg_fragment_mrr",
            "avg_mrr",
            "avg_retrieval_fact_coverage",
        )
    )


def config_distance(left: dict, right: dict, ranges: dict[str, float]) -> float:
    """Mean normalized distance; categorical and missing values use 0/1."""
    keys = left.keys() | right.keys()
    if not keys:
        return 0.0
    distance = 0.0
    for key in keys:
        a, b = left.get(key), right.get(key)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            span = ranges.get(key, 0)
            distance += abs(a - b) / span if span else 0.0
        else:
            distance += float(a != b)
    return distance / len(keys)


def select_fast_candidates(results: list[dict], budget: int) -> list[dict]:
    """Keep the best half, then diverse competitive configs within a fixed budget.

    Equal retrieval scores use fragment completeness and MRR before parameter
    distance. Duplicate configurations cannot spend multiple judge slots.
    """
    if budget <= 0:
        return []
    ranked = sorted(results, key=retrieval_priority, reverse=True)
    unique = {}
    for result in ranked:
        unique.setdefault(json.dumps(result["config"], sort_keys=True), result)
    ranked = list(unique.values())
    if len(ranked) <= budget:
        return ranked
    elite_count = max(1, math.ceil(budget * FAST_ELITE_FRACTION))
    selected = ranked[:elite_count]
    remaining = ranked[elite_count:]
    keys = set().union(*(result["config"].keys() for result in ranked))
    ranges = {}
    for key in keys:
        values = [r["config"].get(key) for r in ranked]
        numeric = [v for v in values if isinstance(v, (int, float))]
        ranges[key] = max(numeric) - min(numeric) if numeric else 0.0
    cutoff = retrieval_priority(ranked[0])[0] * FAST_COMPETITIVE_SCORE_RATIO
    while len(selected) < budget and remaining:
        competitive = [r for r in remaining if retrieval_priority(r)[0] >= cutoff]
        if competitive:
            chosen = max(
                competitive,
                key=lambda r: (
                    min(config_distance(r["config"], s["config"], ranges) for s in selected),
                    retrieval_priority(r),
                ),
            )
        else:
            chosen = remaining[0]
        selected.append(chosen)
        # Dictionaries with identical metrics/configs are value-equal. Only
        # remove the selected object, preserving every other diagnostic row.
        remaining = [r for r in remaining if r is not chosen]
    return selected
