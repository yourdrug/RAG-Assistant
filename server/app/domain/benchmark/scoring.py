"""Metric scales and explicit objective weights for benchmark sweeps."""

import math


METRIC_SCALES = {
    "hit_rate": 1.0,
    "mrr": 1.0,
    "faithfulness": 10.0,
    "relevancy": 10.0,
    "correctness": 10.0,
    "fragment_recall_at_k": 1.0,
    "fragment_mrr": 1.0,
    "context_fragment_recall": 1.0,
    "context_fact_coverage": 1.0,
    "retrieval_fact_coverage": 1.0,
}
DEFAULT_OBJECTIVE_WEIGHTS = {
    "hit_rate": 0.2,
    "mrr": 0.0,
    "faithfulness": 0.4,
    "relevancy": 0.1,
    "correctness": 0.3,
    "fragment_recall_at_k": 0.0,
    "fragment_mrr": 0.0,
    "context_fragment_recall": 0.0,
    "context_fact_coverage": 0.0,
    "retrieval_fact_coverage": 0.0,
}
RETRIEVAL_OBJECTIVES = frozenset(
    {"hit_rate", "mrr", "fragment_recall_at_k", "fragment_mrr", "retrieval_fact_coverage"}
)


def validate_objective_weights(
    weights: dict[str, float], *, require_retrieval: bool = True
) -> dict[str, float]:
    """Omitted metrics have zero weight; never introduce hidden objectives."""
    if set(weights) - METRIC_SCALES.keys():
        msg = "Unknown objective metric"
        raise ValueError(msg)

    resolved = dict.fromkeys(METRIC_SCALES, 0.0)

    for metric, value in weights.items():
        weight = float(value)

        if not math.isfinite(weight) or weight < 0:
            msg = "Objective weights must be finite and non-negative"
            raise ValueError(msg)

        resolved[metric] = weight

    if not any(resolved.values()):
        msg = "At least one objective weight must be positive"
        raise ValueError(msg)

    if require_retrieval and not any(resolved[key] > 0 for key in RETRIEVAL_OBJECTIVES):
        msg = "At least one retrieval objective weight must be positive"
        raise ValueError(msg)

    return resolved
