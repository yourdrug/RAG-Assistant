"""Validate v2 structure, identities and legacy compatibility, without running RAG."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any


BEHAVIORS = {
    "factual": "answer", "negative": "answer", "false_premise": "correct_premise",
    "not_in_corpus": "abstain", "out_of_scope": "abstain", "ambiguous": "clarify_ambiguity",
}
DEFAULT_PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "benchmark-questions-v2.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_cases(cases: Any) -> None:
    require(isinstance(cases, list), "Expected top-level JSON list")
    identities: dict[str, str] = {}
    datasets: dict[str, set[str]] = defaultdict(set)
    for index, case in enumerate(cases, 1):
        require(isinstance(case, dict), f"{index}: expected object")
        for key in ("question", "expected_answer", "source_hint", "annotations", "tags", "dataset", "is_active", "case_id"):
            require(key in case, f"{index}: missing {key}")
        annotations = case["annotations"]
        require(isinstance(annotations, dict), f"{index}: annotations must be object")
        kind = annotations.get("answer_type")
        require(isinstance(kind, str) and kind in BEHAVIORS, f"{index}: invalid answer_type")
        require(annotations.get("expected_behavior") == BEHAVIORS[kind], f"{index}: behavior mismatch")
        require(annotations.get("expected_refusal") is (kind in ("not_in_corpus", "out_of_scope")), f"{index}: legacy mismatch")
        require(isinstance(case["is_active"], bool), f"{index}: is_active must be bool")
        require(isinstance(annotations.get("source_files"), list), f"{index}: source_files must be list")
        require(annotations.get("evidence_text_check") in ("matched", "unmatched", "not_provided"), f"{index}: invalid evidence_text_check")
        require(isinstance(annotations.get("review_required"), bool), f"{index}: review_required must be bool")
        identity = case["case_id"]
        require(isinstance(identity, str) and bool(identity.strip()), f"{index}: invalid case_id")
        require(isinstance(case["dataset"], str), f"{index}: invalid dataset")
        require(identity not in identities or identities[identity] == case["question"], f"{index}: case_id reused for different questions")
        require(case["dataset"] not in datasets[identity], f"{index}: case_id repeated within dataset")
        identities[identity] = case["question"]
        datasets[identity].add(case["dataset"])
    for case in cases:
        if case.get("variant_of") is not None:
            require(case["variant_of"] in identities, f"{case['case_id']}: variant_of not found")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args()
    cases = json.loads(args.path.read_text(encoding="utf-8"))
    try:
        validate_cases(cases)
    except ValueError as exc:
        parser.error(str(exc))
    active = [case for case in cases if case["dataset"] == "main" and case["is_active"]]
    print(f"OK: {len(cases)} records; {len({c['case_id'] for c in cases})} unique IDs; {len(active)} active main tests")
    print("Categories (main only):", dict(Counter(c["annotations"]["answer_type"] for c in cases if c["dataset"] == "main")))
    print("Manual review:", [c["case_id"] for c in cases if c["annotations"]["review_required"]])


if __name__ == "__main__":
    main()
