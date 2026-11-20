"""Explicit legal act numbers in a question and in document titles.

Pure text rules: an act named in the question ("постановление МНС N 11",
"Закон N 113-З") identifies the primary source of the answer. Matching is by
the act number only, normalised the way source file names store it
("N 11/49/30/26" is stored as "N 11493026").
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
import re
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Callable


ACT_NUMBER_PATTERN = r"(?P<mark>\bN|№)\s*(?P<num>\d+(?:/\d+)*(?:-[ЗзZz])?)(?![\w/-])"
ACT_NUMBER_RE = re.compile(ACT_NUMBER_PATTERN)
ACT_TYPE_PATTERN = r"\b(?:закон|постановлени|инструкци|указ|декрет|приказ|решени|кодекс|положени|правил)\w*"
ACT_MENTION_RE = re.compile(
    rf"(?P<label>(?:{ACT_TYPE_PATTERN}[^,;?()]{{0,60}}?)?{ACT_NUMBER_PATTERN})"
    r"(?P<tail>(?:\s+от\s+\d{1,2}\.\d{1,2}\.\d{4}(?:\s*г\.)?)?"
    r"(?:\s+(?:о|об)\s+[^,;?()]*?)?)"
    r"(?=\s+и\s|\s*[,;?()]|\s*$)",
    re.IGNORECASE,
)
# "пункт № 3", "приложение N 2", "ст. № 5": a numbered provision, not an act.
PROVISION_BEFORE_RE = re.compile(
    r"(?:\b(?:под)?пункт\w*|\bстать\w*|\bчаст\w*|\bглав\w*|\bприложени\w*|\bабзац\w*"
    r"|\bраздел\w*|\bтаблиц\w*|\bформ\w*|\b(?:пп|п|ст|ч|гл|абз|прил)\.?)\s*$",
    re.IGNORECASE,
)
LEADING_LINK_RE = re.compile(r"(?:\s+и)?(?:\s+(?:в|во|по|согласно|из))?\s*$", re.IGNORECASE)
TITLE_FIELDS = ("doc_title", "filename", "source")
MAX_ACT_QUERIES = 3


@dataclass(frozen=True)
class ActMention:
    number: str
    label: str
    start: int
    end: int


def canonical_act_number(raw: str) -> str:
    number = raw.replace("/", "").upper()
    return number.replace("Z", "З")


def names_provision(text: str, match: re.Match[str]) -> bool:
    return PROVISION_BEFORE_RE.search(text[: match.start("mark")]) is not None


def extract_act_numbers(text: str) -> tuple[str, ...]:
    """Distinct act numbers in order of appearance."""
    acts = (m for m in ACT_NUMBER_RE.finditer(text) if not names_provision(text, m))
    return tuple(dict.fromkeys(canonical_act_number(m["num"]) for m in acts))


def act_mentions(question: str) -> list[ActMention]:
    mentions: list[ActMention] = []

    for match in ACT_MENTION_RE.finditer(question):
        if names_provision(question, match):
            continue

        number = canonical_act_number(match["num"])

        if all(mention.number != number for mention in mentions):
            label = " ".join(f"{match['label']}{match['tail']}".split())
            mentions.append(ActMention(number, label, match.start(), match.end()))

    return mentions


def document_act_numbers(metadata: dict[str, Any]) -> frozenset[str]:
    numbers: set[str] = set()

    for field in TITLE_FIELDS:
        value = metadata.get(field)

        if isinstance(value, str) and value:
            numbers.update(extract_act_numbers(PurePosixPath(value.replace("\\", "/")).name))

    return frozenset(numbers)


def referenced_act(metadata: dict[str, Any], numbers: tuple[str, ...]) -> str | None:
    """The first question act number this document is titled with, if any."""
    titled = document_act_numbers(metadata)
    return next((number for number in numbers if number in titled), None)


def cites_referenced_point(metadata: dict[str, Any], points: set[str], acts: tuple[str, ...]) -> bool:
    """Whether a chunk holds a point the question cites, in the act it names (if it names one).

    A chunk can hold several points while its section names only the first, so
    the packed ``point_numbers`` are compared rather than the section heading.
    """
    held = {str(number) for number in metadata.get("point_numbers") or ()}

    if not points & held:
        return False

    return not acts or referenced_act(metadata, acts) is not None


# A multi-word proper term inside a sentence («... центром Государственной системы управления ...»).
PROPER_TERM_RE = re.compile(r"(?<=[а-яё,;]\s)[А-ЯЁ][а-яё]+(?:\s+[а-яё]+){3,}")
BRIDGE_TERM_WORDS = 4


def content_stems(text: str) -> list[str]:
    return [word[:5] for word in re.findall(r"[а-яё]{4,}", text.casefold())]


def proper_terms(text: str) -> set[tuple[str, ...]]:
    """Leading stems of the multi-word proper terms a text uses, inflection-insensitive."""
    terms = (content_stems(match[0]) for match in PROPER_TERM_RE.finditer(text))
    return {tuple(stems[:BRIDGE_TERM_WORDS]) for stems in terms if len(stems) >= BRIDGE_TERM_WORDS}


def compared_act_bridge(
    window: list[tuple[dict[str, Any], str]], acts: tuple[str, ...]
) -> Callable[[dict[str, Any], str], bool]:
    """Whether a chunk of one compared act uses a proper term of another compared act's top chunks.

    Relating two named acts often runs through a term one act uses and the
    other defines; the defining provision rarely ranks high for the question.
    """
    terms: dict[str, set[tuple[str, ...]]] = {}

    if len(acts) > 1:
        for metadata, text in window:
            act = referenced_act(metadata, acts)

            if act is not None:
                terms.setdefault(act, set()).update(proper_terms(text))

    def bridges(metadata: dict[str, Any], text: str) -> bool:
        act = referenced_act(metadata, acts) if terms else None

        if act is None:
            return False

        foreign = {term for other, found in terms.items() if other != act for term in found}
        stems = content_stems(text)
        return any(
            tuple(stems[i : i + BRIDGE_TERM_WORDS]) in foreign
            for i in range(len(stems) - BRIDGE_TERM_WORDS + 1)
        )

    return bridges


def act_focused_queries(question: str) -> list[str]:
    """One query per named act when a question compares two or more acts.

    The shared topic is kept and every other act mention is removed, so one
    salient act cannot dominate candidate generation for the whole question.
    """
    mentions = act_mentions(question)

    if len(mentions) < 2:  # noqa: PLR2004 - a comparison needs two acts
        return []

    topic = question
    previous_type = ""

    for mention in reversed(mentions):
        head = LEADING_LINK_RE.sub("", topic[: mention.start])
        topic = head + topic[mention.end :]

    topic = " ".join(topic.split()).rstrip(" ?.,;")
    queries = []

    for mention in mentions:
        label = mention.label
        typed = re.match(rf"(?:{ACT_TYPE_PATTERN}[^,;?()]*?)(?=(?:\bN|№))", label, re.IGNORECASE)

        if typed:
            previous_type = typed.group().strip()
        elif previous_type:
            label = f"{previous_type} {label}"

        queries.append(f"{topic}: {label}" if topic else label)

    return queries[:MAX_ACT_QUERIES]
