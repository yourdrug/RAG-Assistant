"""LegalDomainProfile -- regulatory acts, laws, contracts.

Content-based splitting by chapter/section/article/clause/sentence.
Versioned: tracks effective dates and act versions.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from domain.ingestion.profiles.base import (
    EFFECTIVE_DATE_TRUST_THRESHOLD_DEFAULT,
    SettingsBackedProfile,
    extract_effective_date_generic,
)
from domain.ingestion.profiles.types import (
    BoundaryLevel,
    ConfigDefault,
    ReferenceMatch,
)


if TYPE_CHECKING:
    from domain.ingestion.profiles.types import (
        EffectiveDateCandidate,
    )


_CHAPTER_RE = re.compile(r"^\s*Глава\s+(\d+[\.\d]*)", re.MULTILINE)
_SECTION_RE = re.compile(r"^\s*Раздел\s+(\d+[\.\d]*)", re.MULTILINE)
_ARTICLE_RE = re.compile(r"^\s*Статья\s+(\d+[\.\d]*)", re.MULTILINE)
_CLAUSE_RE = re.compile(r"^\s*(\d+\.\d+[\.\d]*)\s+", re.MULTILINE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

_ARTICLE_REF_RE = re.compile(r"ст(?:ать[июяе]|\.)\s*(\d+[\.\d]*)", re.IGNORECASE)
_CHAPTER_REF_RE = re.compile(r"глав(?:[ауы]|ой)\s*(\d+[\.\d]*)", re.IGNORECASE)
_LAW_NAME_RE = re.compile(
    r"(?:Федеральный закон|Федеральный конституционный закон|ГК "
    r"РФ|\u041d\u041a РФ|КоАП|\u0423\u041a РФ|\u0422\u041a РФ)"
    r"\s*(?:от\s+[\d.]+\s*)?(?:№\s*[\d]+(?:-[\d\u0410-ЯЁ]+)*)?",
    re.IGNORECASE,
)
_LAW_NUMBER_RE = re.compile(
    r"^\s*(?:Федеральный закон|Федеральный конституционный закон)"
    r"\s*(?:от\s+(?P<date>[^№N\n]+?)\s*)?(?:№|N)\s*(?P<number>[\d]+(?:-[\d\u0410-ЯЁ]+)*)",
    re.IGNORECASE | re.MULTILINE,
)

_EFFECTIVE_PATTERNS = [
    (re.compile(r"вступает в силу\s+(?:\u0441\u043e дня|\u0441)\s+([^\n,]+)", re.IGNORECASE), 0.9),
    (re.compile(r"вступлени[ея]\s+в\s+силу\s+([^\n,]+)", re.IGNORECASE), 0.7),
    (re.compile(r"\u0441\s+даты\s+([^\n,]+)", re.IGNORECASE), 0.5),
]

_SIGNING_DATE_RE = re.compile(r"(\d{1,2}\s+\S+\s+\d{4}\s*\u0433?\.?)")


class LegalDomainProfile(SettingsBackedProfile):
    key = "legal"
    display_name = "Нормативные акты"
    is_versioned = True

    def config_defaults(self) -> list[ConfigDefault]:
        return [
            ConfigDefault(
                "max_unit_chars",
                "1500",
                "int",
                "Safety-net: above this size, split to finer boundary level",
                200,
                10000,
            ),
            ConfigDefault(
                "fingerprint_min_articles",
                "2",
                "int",
                "Minimum article matches for fingerprint detection",
                0,
                20,
            ),
            ConfigDefault(
                "classification_threshold",
                "1.5",
                "float",
                "Density-score threshold when fingerprint doesn't fire",
                0.0,
                20.0,
            ),
            EFFECTIVE_DATE_TRUST_THRESHOLD_DEFAULT,
        ]

    def structural_fingerprint(self, text: str) -> bool:
        min_articles = int(self.get_setting("fingerprint_min_articles"))
        has_articles = len(_ARTICLE_RE.findall(text)) >= min_articles
        has_chapters = bool(_CHAPTER_RE.search(text))
        return has_articles and has_chapters

    def classify_score(self, text: str) -> float:
        score = 0.0
        score += 0.5 * len(_ARTICLE_RE.findall(text))
        score += 0.3 * len(_CHAPTER_RE.findall(text))
        score += 0.2 * len(_SECTION_RE.findall(text))

        if re.search(r"Федеральный закон|ГК РФ|\u041d\u041a РФ", text, re.IGNORECASE):
            score += 2.0

        if re.search(
            r"стороны\s+договорились|настоящ(им|\u0435\u0433\u043e|ий)\s+(договор|соглашени)",
            text,
            re.IGNORECASE,
        ):
            score += 1.5

        return score

    def content_boundaries(self) -> list[BoundaryLevel]:
        return [
            BoundaryLevel("chapter", _CHAPTER_RE),
            BoundaryLevel("section", _SECTION_RE),
            BoundaryLevel("article", _ARTICLE_RE),
            BoundaryLevel("clause", _CLAUSE_RE),
            BoundaryLevel("sentence", _SENTENCE_RE),
        ]

    def extract_references(self, text: str) -> list[ReferenceMatch]:
        refs = [ReferenceMatch("article", m.group(1)) for m in _ARTICLE_REF_RE.finditer(text)]
        refs.extend(ReferenceMatch("chapter", m.group(1)) for m in _CHAPTER_REF_RE.finditer(text))
        refs.extend(ReferenceMatch("law_name", m.group(0).strip()) for m in _LAW_NAME_RE.finditer(text))

        if act_match := _LAW_NUMBER_RE.search(text[:1500]):
            refs.append(ReferenceMatch("act_number", act_match.group("number").upper()))

            if act_match.group("date"):
                refs.append(ReferenceMatch("act_date", act_match.group("date").strip()))

        return refs

    def extract_effective_date(self, text: str) -> EffectiveDateCandidate | None:
        return extract_effective_date_generic(text, _EFFECTIVE_PATTERNS, _SIGNING_DATE_RE)
