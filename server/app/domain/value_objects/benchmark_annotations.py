"""Validate portable benchmark annotations without framework dependencies."""

from copy import deepcopy

ANNOTATION_FIELDS = frozenset(
    {"expected_fragments", "required_facts", "required_conditions", "expected_refusal"}
)
FRAGMENT_FIELDS = frozenset({"source", "text", "pages", "section"})


def _nonempty(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_annotations(value: dict | None) -> dict | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) - ANNOTATION_FIELDS:
        raise ValueError("annotations must be an object containing supported evaluation fields")
    if "expected_refusal" in value and type(value["expected_refusal"]) is not bool:
        raise ValueError("expected_refusal must be a boolean")
    _validate_fragments(value.get("expected_fragments", []))
    for field in ("required_facts", "required_conditions"):
        groups = value.get(field, [])
        if not isinstance(groups, list) or len(groups) > 100:
            raise ValueError(f"{field} must be a list of at most 100 requirements")
        for group in groups:
            alternatives = [group] if isinstance(group, str) else group
            if (
                not isinstance(alternatives, list)
                or not alternatives
                or not all(map(_nonempty, alternatives))
            ):
                raise ValueError(f"{field} entries must be nonempty phrases or lists of alternative phrases")
    return deepcopy(value)


def _validate_fragments(fragments: list) -> None:
    if not isinstance(fragments, list) or len(fragments) > 100:
        raise ValueError("expected_fragments must be a list of at most 100 fragments")
    for fragment in fragments:
        if not isinstance(fragment, dict) or set(fragment) - FRAGMENT_FIELDS:
            raise ValueError("each expected fragment must contain only source, text, pages and section")
        if not _nonempty(fragment.get("source")) or not _nonempty(fragment.get("text")):
            raise ValueError("each expected fragment requires nonempty source and text")
        if "section" in fragment and not _nonempty(fragment["section"]):
            raise ValueError("fragment section must be nonempty")
        if "pages" in fragment:
            pages = fragment["pages"]
            if not isinstance(pages, list) or not pages or any(type(p) is not int or p < 1 for p in pages):
                raise ValueError("fragment pages must be a nonempty list of positive integers")
