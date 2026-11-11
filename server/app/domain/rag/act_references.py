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
from typing import Any


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
