"""Benchmark questions and bounded, validated evidence annotations."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from domain.shared.exceptions import ValidationError


if TYPE_CHECKING:
    from datetime import date


DEFAULT_DATASET = "main"


MAX_ANNOTATION_ITEMS = 100


ANNOTATION_FIELDS = frozenset(
    {"expected_fragments", "required_facts", "required_conditions", "expected_refusal"}
)
FRAGMENT_FIELDS = frozenset({"source", "text", "pages", "section", "document_id", "act_version_id"})


def nonempty(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_annotations(value: object) -> dict[str, Any] | None:
    if value is None:
        return None

    if not isinstance(value, dict) or set(value) - ANNOTATION_FIELDS:
        msg = "annotations must be an object containing supported evaluation fields"
        raise ValidationError(msg)

    if "expected_refusal" in value and type(value["expected_refusal"]) is not bool:
        msg = "expected_refusal must be a boolean"
        raise ValidationError(msg)

    validate_fragments(value.get("expected_fragments", []))

    for field_name in ("required_facts", "required_conditions"):
        groups = value.get(field_name, [])

        if not isinstance(groups, list) or len(groups) > MAX_ANNOTATION_ITEMS:
            msg = f"{field_name} must be a list of at most 100 requirements"
            raise ValidationError(msg)

        for group in groups:
            alternatives = [group] if isinstance(group, str) else group

            if not isinstance(alternatives, list) or not alternatives or not all(map(nonempty, alternatives)):
                msg = f"{field_name} entries must be nonempty phrases or lists of alternative phrases"
                raise ValidationError(msg)

    return deepcopy(value)


def validate_fragment_alternatives(alternatives: object) -> None:
    if (
        not isinstance(alternatives, list)
        or not alternatives
        or any(not isinstance(item, dict) or "alternatives" in item for item in alternatives)
    ):
        message = "fragment alternatives must be a nonempty list of source fragments"
        raise ValidationError(message)


def validate_fragments(fragments: object) -> None:
    if not isinstance(fragments, list) or len(fragments) > MAX_ANNOTATION_ITEMS:
        msg = "expected_fragments must be a list of at most 100 fragments"
        raise ValidationError(msg)

    for fragment in fragments:
        if isinstance(fragment, dict) and set(fragment) == {"alternatives"}:
            alternatives = fragment["alternatives"]
            validate_fragment_alternatives(alternatives)
            validate_fragments(alternatives)
            continue

        validate_fragment(fragment)


def validate_fragment(fragment: object) -> None:
    if not isinstance(fragment, dict) or set(fragment) - FRAGMENT_FIELDS:
        msg = "each expected fragment must contain supported source, text, location and identity fields"
        raise ValidationError(msg)

    if not nonempty(fragment.get("source")) or not nonempty(fragment.get("text")):
        msg = "each expected fragment requires nonempty source and text"
        raise ValidationError(msg)

    if "section" in fragment and not nonempty(fragment["section"]):
        msg = "fragment section must be nonempty"
        raise ValidationError(msg)

    if "pages" in fragment:
        pages = fragment["pages"]

        if not isinstance(pages, list) or not pages or any(type(p) is not int or p < 1 for p in pages):
            msg = "fragment pages must be a nonempty list of positive integers"
            raise ValidationError(msg)


@dataclass
class BenchmarkQuestion:
    question: str = ""
    expected_answer: str | None = None
    source_hint: str | None = None
    annotations: dict[str, Any] | None = None
    tags: list[str] | None = None
    dataset: str = DEFAULT_DATASET
    is_active: bool = True
    created_by: int | None = None
    notes: str | None = None
    history: list[dict[str, str]] | None = None
    as_of_date: date | None = None
    id: int | None = None
    creation_date: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        self.annotations = validate_annotations(self.annotations)
        self.history = validate_history(self.history)


def validate_history(value: object) -> list[dict[str, str]] | None:
    if value is None:
        return None

    if not isinstance(value, list) or len(value) > MAX_ANNOTATION_ITEMS:
        message = "history must be a list of at most 100 messages"
        raise ValidationError(message)

    for item in value:
        if (
            not isinstance(item, dict)
            or set(item) != {"role", "content"}
            or item.get("role") not in {"user", "assistant"}
            or not isinstance(item.get("content"), str)
        ):
            message = "history messages require user/assistant role and string content"
            raise ValidationError(message)

    return deepcopy(value)
