"""Pure comparison of persisted benchmark summaries for web regression checks."""

DEFAULT_THRESHOLDS = {
    "hit_rate": -0.05,
    "faithfulness": -1.0,
    "relevancy": -1.0,
    "correctness": -1.0,
    "avg_context_precision": -1.0,
    "avg_context_recall": -1.0,
}


def compare_runs(
    current: dict,
    baseline: dict,
    thresholds: dict | None = None,
) -> dict:
    """Compare current run against baseline.

    Returns dict with:
      - passed: bool (True if all metrics within thresholds)
      - results: list of dicts with metric_name, baseline, current, delta, threshold, failed
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    current_metrics = current.get("metrics", {})
    baseline_metrics = baseline.get("metrics", {})

    results = []
    for metric_name, threshold in thresholds.items():
        curr_val = current_metrics.get(metric_name)
        base_val = baseline_metrics.get(metric_name)

        if curr_val is None or base_val is None:
            results.append(
                {
                    "metric": metric_name,
                    "baseline": base_val,
                    "current": curr_val,
                    "delta": None,
                    "threshold": threshold,
                    "failed": False,
                    "note": "skipped (missing data)",
                }
            )
            continue

        delta = curr_val - base_val
        failed = delta < threshold

        results.append(
            {
                "metric": metric_name,
                "baseline": round(base_val, 4),
                "current": round(curr_val, 4),
                "delta": round(delta, 4),
                "threshold": threshold,
                "failed": failed,
            }
        )

    all_passed = not any(r["failed"] for r in results)
    return {"passed": all_passed, "results": results}
