"""Shared numeric, letter and sentence boundaries for regulatory document profiles."""

import re


POINT_RE = re.compile(r"^\s*(\d+)\.\s+(?=\S)", re.MULTILINE)
SUBPOINT_NUM_RE = re.compile(r"^\s*(\d+\.\d+(?:\.\d+)*)\.?\s+", re.MULTILINE)
DELIMITED_SUBPOINT_NUM_RE = re.compile(r"^\s*(\d+\.\d+(?:\.\d+)*)\.\s+", re.MULTILINE)
SUBPOINT_LETTER_RE = re.compile(r"^\s*([а-яё])\)\s+", re.MULTILINE)
PARAGRAPH_RE = re.compile(r"\n\s*\n")


def regulatory_unit_label(kind: str, value: str) -> str:
    """Shared Russian labels for legislation and decrees."""
    labels = {
        "chapter": "Глава",
        "section": "Раздел",
        "article": "Статья",
        "point": "п.",
        "subpoint_num": "пп.",
        "subpoint": "подп.",
    }
    label = labels.get(kind)
    return f"{label} {value}{')' if kind == 'subpoint' else ''}" if label else ""


# Periods of abbreviations common in legal texts never end a sentence: "в ред. Закона",
# "ст. 5", "т.е.". Single letters cover initials and "п.", "ч.", "г.". Python lookbehind
# is fixed-width, so each abbreviation gets its own negative lookbehind.
LEGAL_ABBREVIATIONS = (
    "ред",
    "изм",
    "доп",
    "утв",
    "ст",
    "пп",
    "подп",
    "гл",
    "разд",
    "абз",
    "гг",
    "руб",
    "коп",
    "тыс",
    "млн",
    "млрд",
    "им",
    "см",
    "напр",
    "т.е",
    "т.п",
    "т.д",
    "т.к",
    "т.ч",
)
# A number ends a sentence ("согласно статье 15."), except where it numbers a
# unit: a line-start item ("1. Текст"), a nested item ("1.2.") or a heading
# ("Статья 1. Термины"). Lookbehinds need fixed widths, hence one per digit count.
NUMBERED_UNIT_WORDS = ("статья", "глава", "раздел", "пункт", "подпункт", "часть", "параграф")
NUMBER_WIDTHS = (1, 2, 3)
SENTENCE_RE = re.compile(
    r"(?<=[.!?])"
    r"(?<!\b[а-яёa-z]\.)"
    + "".join(rf"(?<!^\d{{{width}}}\.)(?<!\.\d{{{width}}}\.)" for width in NUMBER_WIDTHS)
    + "".join(rf"(?<!\b{word} \d{{{width}}}\.)" for word in NUMBERED_UNIT_WORDS for width in NUMBER_WIDTHS)
    + "".join(rf"(?<!\b{re.escape(abbreviation)}\.)" for abbreviation in LEGAL_ABBREVIATIONS)
    + r"\s+",
    re.IGNORECASE | re.MULTILINE,
)
