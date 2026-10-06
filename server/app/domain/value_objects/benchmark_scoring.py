"""Metric scales and explicit objective weights for benchmark sweeps."""

import math

METRIC_SCALES = {
    "hit_rate": 1.0,
    "mrr": 1.0,
    "faithfulness": 10.0,
    "relevancy": 10.0,
    "correctness": 10.0,
}
DEFAULT_OBJECTIVE_WEIGHTS = {
    "hit_rate": 0.4,
    "mrr": 0.0,
    "faithfulness": 0.3,
    "relevancy": 0.3,
    "correctness": 0.0,
}


def validate_objective_weights(weights: dict[str, float]) -> dict[str, float]:
    """Omitted metrics have zero weight; never introduce hidden objectives."""
    if set(weights) - METRIC_SCALES.keys():
        raise ValueError("Unknown objective metric")
    resolved = dict.fromkeys(METRIC_SCALES, 0.0)
    for metric, value in weights.items():
        weight = float(value)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("Objective weights must be finite and non-negative")
        resolved[metric] = weight
    if not any(resolved.values()):
        raise ValueError("At least one objective weight must be positive")
    if resolved["hit_rate"] + resolved["mrr"] <= 0:
        raise ValueError("At least one retrieval weight (hit_rate or mrr) must be positive")
    return resolved
