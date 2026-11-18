"""Whether the expected fragments of benchmark questions exist whole inside one chunk of the corpus.

A fragment that no chunk contains cannot be retrieved by any search setting, so its question
measures annotation or ingestion, not retrieval.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from domain.benchmark.evaluation import matches_fragment, source_name
from domain.benchmark.question import evaluation_annotations


# Long fragments are shortened in reports; the full text stays in the question.
REPORT_TEXT_LENGTH = 160


class UnreachableReason(StrEnum):
    NO_DOCUMENT = "no_document"
    NOT_IN_CHUNK = "not_in_chunk"


@dataclass(frozen=True)
class UnreachableFragment:
    question_id: int | str | None
    question: str
    source: str
    text: str
    reason: UnreachableReason

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "source": self.source,
            "text": self.text,
            "reason": self.reason.value,
        }


@dataclass(frozen=True)
class ReachabilityReport:
    questions: int
    fragments: int
    reachable: int
    unreachable: list[UnreachableFragment] = field(default_factory=list)

    @property
    def share(self) -> float | None:
        return round(self.reachable / self.fragments, 4) if self.fragments else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "questions": self.questions,
            "fragments": self.fragments,
            "reachable": self.reachable,
            "share": self.share,
            "unreachable": [item.to_dict() for item in self.unreachable],
        }


def expected_fragments(annotations: dict[str, Any] | None) -> list[dict[str, Any]]:
    return list((evaluation_annotations(annotations) or {}).get("expected_fragments") or [])


def fragment_sources(fragment: dict[str, Any]) -> set[str]:
    """Normalized file names a fragment, or any of its alternatives, may come from."""
    options = fragment.get("alternatives") or [fragment]
    return {source_name(option["source"]) for option in options if option.get("source")}


def fragment_reachability(
    questions: list[dict[str, Any]], documents: list[dict[str, Any]]
) -> ReachabilityReport:
    """Questions carry `id`, `question` and `annotations`; documents are retrieval-shaped chunks."""
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for document in documents:
        by_source[source_name(document["metadata"].get("source", ""))].append(document)

    labelled = total = reachable = 0
    unreachable = []

    for question in questions:
        fragments = expected_fragments(question.get("annotations"))
        labelled += bool(fragments)

        for fragment in fragments:
            total += 1
            candidates = [doc for source in fragment_sources(fragment) for doc in by_source.get(source, [])]

            if candidates and any(matches_fragment(fragment, doc) for doc in candidates):
                reachable += 1
                continue

            first = (fragment.get("alternatives") or [fragment])[0]
            unreachable.append(
                UnreachableFragment(
                    question_id=question.get("id"),
                    question=question.get("question", ""),
                    source=first.get("source", ""),
                    text=first.get("text", "")[:REPORT_TEXT_LENGTH],
                    reason=UnreachableReason.NOT_IN_CHUNK if candidates else UnreachableReason.NO_DOCUMENT,
                )
            )

    return ReachabilityReport(
        questions=labelled, fragments=total, reachable=reachable, unreachable=unreachable
    )
