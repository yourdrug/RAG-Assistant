"""Recalculate stage diagnostics and compare saved runs without any model calls."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from domain.benchmark.evaluation import evaluate_evidence, evaluate_stage_evidence
from domain.benchmark.stage_reporting import compare_stage_runs, summarize_stage_tags, diagnostic_card


def saved_cases(value: object) -> list[dict]:
    if isinstance(value, list):
        if all(isinstance(item, dict) and "question" in item for item in value):
            return value
        if len(value) != 1:
            raise ValueError("Select exactly one run with --run-id / --baseline-run-id")
        return saved_cases(value[0])
    if isinstance(value, dict):
        for key in ("per_question_results", "results"):
            if key in value:
                cases = saved_cases(value[key])
                return [{"dataset": value.get("dataset") or value.get("run", {}).get("dataset"), **case} for case in cases]
    raise ValueError("Expected saved case results or exported benchmark runs")


def replay_cases(cases: list[dict]) -> list[dict]:
    results = deepcopy(cases)
    for case in results:
        evidence = case.get("evidence") or {}
        annotations = case.get("annotations")
        validation = evidence.get("annotation_validation") or case.get("annotation_validation") or {}
        if validation.get("status") == "invalid_annotation":
            annotations = None
        scores, diagnostics = evaluate_evidence(
            annotations, evidence.get("retrieved"), evidence.get("selected"),
            evidence.get("context"), case.get("answer") or "",
            judge_scores=case.get("evidence_metrics"),
        )
        case["evidence_metrics"] = scores
        case["evidence_diagnostics"] = {**case.get("evidence_diagnostics", {}), **diagnostics}
        case["stage_evaluation"] = evaluate_stage_evidence(case.get("annotations"), evidence, case.get("answer") or "")
        case["diagnostic_card"] = diagnostic_card(case)
    return results


def select_run(value: object, run_id: int | None) -> object:
    if run_id is None:
        return value
    runs = value if isinstance(value, list) else [value]
    selected = [run for run in runs if run.get("id") == run_id or run.get("run", {}).get("id") == run_id]
    if len(selected) != 1:
        raise ValueError("Expected exactly one matching run ID")
    return selected[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--run-id", type=int)
    parser.add_argument("--baseline-run-id", type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cases = replay_cases(saved_cases(select_run(json.loads(args.input.read_text()), args.run_id)))
    report = {"results": cases, "summary": summarize_stage_tags(cases), "judge_reevaluated": False}
    if args.baseline:
        baseline = replay_cases(saved_cases(select_run(json.loads(args.baseline.read_text()), args.baseline_run_id)))
        report["comparison"] = compare_stage_runs(baseline, cases)
        tags = sorted({tag for case in cases for tag in case.get("tags") or []})
        report["comparison_by_tags"] = {tag: compare_stage_runs(
            [case for case in baseline if tag in (case.get("tags") or [])],
            [case for case in cases if tag in (case.get("tags") or [])],
        ) for tag in tags}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
