"""Benchmark question: domain models and rules."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime

DEFAULT_DATASET = "main"


ANNOTATION_FIELDS = frozenset(
    {"expected_fragments", "required_facts", "required_conditions", "expected_refusal"}
)
FRAGMENT_FIELDS = frozenset({"source", "text", "pages", "section"})


def nonempty(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_annotations(value: dict | None) -> dict | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) - ANNOTATION_FIELDS:
        raise ValueError("annotations must be an object containing supported evaluation fields")
    if "expected_refusal" in value and type(value["expected_refusal"]) is not bool:
        raise ValueError("expected_refusal must be a boolean")
    validate_fragments(value.get("expected_fragments", []))
    for field_name in ("required_facts", "required_conditions"):
        groups = value.get(field_name, [])
        if not isinstance(groups, list) or len(groups) > 100:
            raise ValueError(f"{field_name} must be a list of at most 100 requirements")
        for group in groups:
            alternatives = [group] if isinstance(group, str) else group
            if not isinstance(alternatives, list) or not alternatives or not all(map(nonempty, alternatives)):
                raise ValueError(
                    f"{field_name} entries must be nonempty phrases or lists of alternative phrases"
                )
    return deepcopy(value)


def validate_fragments(fragments: list) -> None:
    if not isinstance(fragments, list) or len(fragments) > 100:
        raise ValueError("expected_fragments must be a list of at most 100 fragments")
    for fragment in fragments:
        if not isinstance(fragment, dict) or set(fragment) - FRAGMENT_FIELDS:
            raise ValueError("each expected fragment must contain only source, text, pages and section")
        if not nonempty(fragment.get("source")) or not nonempty(fragment.get("text")):
            raise ValueError("each expected fragment requires nonempty source and text")
        if "section" in fragment and not nonempty(fragment["section"]):
            raise ValueError("fragment section must be nonempty")
        if "pages" in fragment:
            pages = fragment["pages"]
            if not isinstance(pages, list) or not pages or any(type(p) is not int or p < 1 for p in pages):
                raise ValueError("fragment pages must be a nonempty list of positive integers")


@dataclass
class BenchmarkQuestion:
    question: str = ""
    expected_answer: str | None = None
    source_hint: str | None = None
    annotations: dict | None = None
    tags: list[str] | None = None
    dataset: str = DEFAULT_DATASET
    is_active: bool = True
    created_by: int | None = None
    notes: str | None = None
    id: int | None = None
    creation_date: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        self.annotations = validate_annotations(self.annotations)
