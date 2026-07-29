"""Benchmark History — track metric trends across runs for regression detection."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

log = logging.getLogger("default")

HISTORY_FILE = "benchmark_history.jsonl"

DEFAULT_THRESHOLDS = {
    "hit_rate": -0.05,
    "faithfulness": -1.0,
    "relevancy": -1.0,
    "correctness": -1.0,
}


def _history_path(data_dir: str) -> Path:
    return Path(data_dir) / HISTORY_FILE


def save_summary_to_history(summary: dict, config: dict, data_dir: str) -> None:
    """Append a run summary to the JSONL history file."""
    metrics = {
        "hit_rate": summary.get("hit_rate"),
        "avg_mrr": summary.get("avg_mrr"),
        "avg_faithfulness": summary.get("avg_faithfulness"),
        "avg_relevancy": summary.get("avg_relevancy"),
        "avg_correctness": summary.get("avg_correctness"),
        "avg_similarity": summary.get("avg_similarity"),
    }

    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": config,
        "metrics": {k: v for k, v in metrics.items() if v is not None},
        "total_questions": summary.get("total_questions", 0),
        "total_time_sec": summary.get("total_time_sec", 0),
    }

    path = _history_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    log.info("Benchmark history saved to %s", path)


def load_history(data_dir: str) -> list[dict]:
    """Load all historical benchmark runs from JSONL."""
    path = _history_path(data_dir)
    if not path.exists():
        return []

    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def get_last_baseline(data_dir: str) -> dict | None:
    """Return the most recent benchmark run as baseline for regression comparison."""
    history = load_history(data_dir)
    return history[-1] if history else None


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


def print_history(data_dir: str, n: int = 10) -> None:
    """Print last N benchmark runs as a table."""
    history = load_history(data_dir)
    if not history:
        log.info("No benchmark history found.")
        return

    recent = history[-n:]
    log.info("=" * 90)
    log.info("BENCHMARK HISTORY  (last %d runs)", len(recent))
    log.info("=" * 90)
    log.info(
        "%-20s  %6s  %6s  %6s  %6s  %6s  %5s",
        "Timestamp",
        "HR",
        "MRR",
        "Faith",
        "Rel",
        "Corr",
        "N",
    )
    log.info("-" * 90)

    for r in recent:
        ts = r.get("timestamp", "?")
        m = r.get("metrics", {})
        log.info(
            "%-20s  %5.1f%%  %.3f  %5.1f  %5.1f  %5s  %3d",
            ts[:19],
            (m.get("hit_rate") or 0) * 100,
            m.get("avg_mrr") or 0,
            m.get("avg_faithfulness") or 0,
            m.get("avg_relevancy") or 0,
            f"{m.get('avg_correctness'):.1f}" if m.get("avg_correctness") is not None else "  -",
            r.get("total_questions", 0),
        )

    log.info("=" * 90)

    if len(history) >= 2:
        latest = history[-1]
        prev = history[-2]
        comp = compare_runs(latest, prev)
        log.info("Change vs previous run:")
        for r in comp["results"]:
            status = "FAIL" if r["failed"] else "ok"
            if r["delta"] is not None:
                log.info(
                    "  %-18s  %s  delta=%+.4f  (threshold=%+.4f)  [%s]",
                    r["metric"],
                    status,
                    r["delta"],
                    r["threshold"],
                    r.get("note", ""),
                )
        if comp["passed"]:
            log.info("  -> PASSED (no regression detected)")
        else:
            log.warning("  -> FAILED (regression detected)")
