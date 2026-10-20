"""Availability and failure diagnostics for independent benchmark judge scores."""


def judge_coverage(results: list[dict]) -> dict:
    """Count available scores and cases without turning failures into zeroes."""
    coverage = {}
    for key in ("faithfulness", "relevancy", "correctness", "context_precision", "context_recall"):
        group = "context_metrics" if key.startswith("context_") else "generator_metrics"
        coverage[f"{key}_evaluated_count"] = sum(r.get(group, {}).get(key) is not None for r in results)
        coverage[f"{key}_expected_count"] = sum(
            bool(r.get("expected_answer")) if key == "correctness" else True for r in results
        )
    failed = sum(bool(result_judge_errors(r)) for r in results)
    coverage.update(judge_evaluated_count=len(results) - failed, judge_error_count=failed)
    return coverage


def result_judge_errors(result: dict) -> dict:
    errors = {}
    for group in ("generator_metrics", "context_metrics"):
        for key, value in result.get(group, {}).items():
            if key.endswith("_error"):
                errors[key.removesuffix("_error")] = value
            elif key.endswith("_reason") and str(value).startswith("[Error:"):
                errors[key.removesuffix("_reason")] = value
    for key, detail in result.get("evidence_diagnostics", {}).get("judge", {}).items():
        if "error" in detail:
            errors[key] = detail["error"]
    return errors
