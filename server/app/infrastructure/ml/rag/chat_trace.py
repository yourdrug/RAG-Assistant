"""Production projection of the shared evidence collector, with a bounded JSON payload."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from decimal import Decimal
import json
import logging
from typing import TYPE_CHECKING, Any

from config import ObservabilitySection, request_section
from domain.rag.outcome import AnswerOutcome
from infrastructure.ml.rag.benchmark_evidence import BenchmarkEvidence, active_evidence, trace_document


if TYPE_CHECKING:
    from collections.abc import Iterator

    from application.chat.ports import PIIRedactorPort

MAX_TRACE_BYTES = 64 * 1024
VERDICT_KEYS = ("relevant", "is_relevant", "sufficient", "is_sufficient")


def fixed_point(value: object) -> object:
    if isinstance(value, float):
        # Quoted, so each number is counted two bytes longer than PostgreSQL prints it.
        return format(Decimal(repr(value)), "f")

    if isinstance(value, dict):
        return {key: fixed_point(item) for key, item in value.items()}

    if isinstance(value, list):
        return [fixed_point(item) for item in value]

    return value


def stored_size(data: dict[str, Any]) -> int:
    """Upper bound of ``jsonb::text`` bytes checked by ``ck_chat_log_trace_size``.

    PostgreSQL prints numbers without an exponent (``1e-05`` becomes ``0.00001``),
    so measuring Python's float repr could pass here and fail the INSERT, rolling
    back the whole answer transaction.
    """
    return len(json.dumps(fixed_point(data), ensure_ascii=False).encode())


def refusal_stage(attempts: list[dict[str, Any]]) -> str:
    """The last gate that returned a negative verdict; otherwise retrieval refused."""
    for attempt in reversed(attempts):
        for check in reversed(attempt["checks"]):
            if any(check["result"].get(key) is False for key in VERDICT_KEYS):
                return str(check["stage"])

    return "retrieval"


def bounded_trace(data: dict[str, Any]) -> dict[str, Any]:
    """Drop stage tails first, preserving an explicit truncation marker."""
    while stored_size(data) > MAX_TRACE_BYTES:
        data["truncated"] = True
        lists = [
            value
            for attempt in data["attempts"]
            for value in [
                attempt["exclusions"],
                *[stage["documents"] for stage in attempt["stages"].values()],
            ]
            if value
        ]

        if lists:
            longest = max(lists, key=len)
            del longest[len(longest) // 2 :]
        elif data["attempts"]:
            data["attempts"].pop()
        else:
            return {"schema_version": 1, "attempts": [], "truncated": True}

    return data


def production_trace(evidence: BenchmarkEvidence, redactor: PIIRedactorPort | None) -> dict[str, Any]:
    def redact(value: str) -> str:
        return redactor.redact(value) if redactor is not None else value

    attempts = []

    for index, attempt in enumerate(evidence.attempts):
        attempts.append(
            {
                "question": redact(attempt["question"]),
                "query": redact(attempt["query"]),
                "sub_queries": [redact(q) for q in attempt["sub_queries"]],
                "kind": attempt.get("kind", "initial" if index == 0 else "refinement"),
                "search": attempt.get("search", {}),
                "stages": {
                    ("retrieved" if name == "candidates" else name): {
                        "status": stage["status"],
                        "ordered": stage["ordered"],
                        "documents": [trace_document(doc) for doc in stage["documents"]],
                        **(
                            {
                                "budget": {
                                    key: value
                                    for key, value in stage["budget"].items()
                                    if key
                                    in {
                                        "num_ctx",
                                        "num_predict",
                                        "input_tokens",
                                        "base_tokens",
                                        "history_tokens",
                                        "context_tokens",
                                        "documents_not_fitted",
                                        "answer_style",
                                        "enumerate_cases",
                                    }
                                }
                            }
                            if stage.get("budget")
                            else {}
                        ),
                    }
                    for name, stage in attempt["stages"].items()
                },
                "exclusions": [trace_document(item) for item in attempt["exclusions"]],
                "checks": [
                    {
                        "stage": check["stage"],
                        "result": {
                            key: redact(value)[:500] if isinstance(value, str) else value
                            for key, value in check["result"].items()
                        },
                    }
                    for check in attempt.get("checks", [])
                ],
            }
        )

    return bounded_trace(
        {
            "schema_version": 1,
            "attempts": attempts,
            "completion": evidence.completion or evidence.trace_data.get("outcome"),
            "refusal_stage": refusal_stage(attempts)
            if evidence.completion == AnswerOutcome.NOT_FOUND
            else None,
            "generation": evidence.completion_details or {"generator_called": False, "finish_reason": None},
        }
    )


@contextmanager
def chat_trace_scope(redactor: PIIRedactorPort | None) -> Iterator[None]:
    if active_evidence.get() is not None or not request_section(ObservabilitySection).trace_enabled:
        yield
        return

    evidence = BenchmarkEvidence(benchmark_mode=False)
    token = active_evidence.set(evidence)
    try:
        yield
    except (asyncio.CancelledError, GeneratorExit):
        evidence.completion = AnswerOutcome.CANCELLED
        raise
    except Exception:
        evidence.completion = AnswerOutcome.ERROR
        raise
    finally:
        try:
            if evidence.completion == AnswerOutcome.INPUT_BLOCKED:
                evidence.trace_data.clear()
            else:
                data = production_trace(evidence, redactor)
                evidence.trace_data.clear()
                evidence.trace_data.update(data)
        except Exception:  # noqa: BLE001 - trace capture must preserve the answer/error
            evidence.trace_data.clear()
            logging.getLogger(__name__).warning("Chat trace capture failed")
        finally:
            active_evidence.reset(token)
