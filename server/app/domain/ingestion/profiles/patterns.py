"""Shared numeric, letter and sentence boundaries for regulatory document profiles."""

import re


POINT_RE = re.compile(r"^\s*(\d+)\.\s+(?=\S)", re.MULTILINE)
SUBPOINT_NUM_RE = re.compile(r"^\s*(\d+\.\d+(?:\.\d+)*)\.?\s+", re.MULTILINE)
DELIMITED_SUBPOINT_NUM_RE = re.compile(r"^\s*(\d+\.\d+(?:\.\d+)*)\.\s+", re.MULTILINE)
SUBPOINT_LETTER_RE = re.compile(r"^\s*([а-яё])\)\s+", re.MULTILINE)
PARAGRAPH_RE = re.compile(r"\n\s*\n")

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
SENTENCE_RE = re.compile(
    r"(?<=[.!?])"
    r"(?<!\b[а-яёa-z]\.)"
    + "".join(rf"(?<!\b{re.escape(abbreviation)}\.)" for abbreviation in LEGAL_ABBREVIATIONS)
    + r"\s+",
    re.IGNORECASE,
)
