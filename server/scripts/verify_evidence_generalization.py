"""Run frozen, unseen evidence probes through production extractors, without an LLM.

Run from server with PYTHONPATH=app. A declined extraction is not proof that
the downstream LLM refuses: this checks the decision to bypass that LLM.
"""

import argparse
import json
import logging
from pathlib import Path
from time import perf_counter

from langchain_core.documents import Document

from infrastructure.ml.rag.evidence_focus import evidence_reading_map
from infrastructure.ml.rag.grounded_response import (
    extractive_procedure_response,
    extractive_table_comparison_response,
    extractive_timing_response,
    missing_field_response,
)

logger = logging.getLogger("default")


def evaluate(case):
    docs = [(Document(**item), 0.9) for item in case.get("docs", [])]
    started = perf_counter()
    if case["kind"] == "timing":
        answer = extractive_timing_response(docs, case["question"])
    elif case["kind"] == "procedure":
        answer = extractive_procedure_response(docs, case["question"])
    elif case["kind"] == "amendment":
        answer = "\n".join(evidence_reading_map(case["context"], case["question"]))
    else:
        answer = missing_field_response(case["context"], case["question"])
        if answer is None:
            answer = extractive_table_comparison_response(case["context"], case["question"])
    elapsed = perf_counter() - started
    checks = {
        "declined": answer is None if case.get("expect_decline") else True,
        "required": all(value in (answer or "") for value in case.get("required", [])),
        "forbidden": all(value not in (answer or "") for value in case.get("forbidden", [])),
    }
    return {
        **case,
        "answer": answer,
        "checks": checks,
        "passed": all(checks.values()),
        "extractor_latency_sec": elapsed,
        "llm_calls": 0,
        "llm_cost_usd": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    snapshot = json.loads(args.input.read_text())
    results = [evaluate(case) for case in snapshot["cases"]]
    output = {
        "scope": "Production extractor probes, no retrieval or downstream LLM",
        "passed": sum(case["passed"] for case in results),
        "total": len(results),
        "results": results,
    }
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    for case in results:
        if not case["passed"]:
            logger.warning("Probe %s failed: %s", case["id"], case["answer"])
    logger.warning("%s/%s probes passed", output["passed"], output["total"])
    if output["passed"] != output["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
