"""LegalDomainProfile -- regulatory acts, laws, contracts.

Content-based splitting by chapter/section/article/clause/sentence.
Versioned: tracks effective dates and act versions.
"""

from __future__ import annotations

import re
from datetime import date

from domain.domain_profile.base import (
    EFFECTIVE_DATE_TRUST_THRESHOLD_DEFAULT,
    SettingsBackedProfile,
    extract_effective_date_generic,
    versioned_prompt_date_stamp,
)
from domain.domain_profile.protocol import (
    BoundaryLevel,
    ConfigDefault,
    EffectiveDateCandidate,
    ReferenceMatch,
)

_CHAPTER_RE = re.compile(r"^\s*Глава\s+(\d+[\.\d]*)", re.MULTILINE)
_SECTION_RE = re.compile(r"^\s*Раздел\s+(\d+[\.\d]*)", re.MULTILINE)
_ARTICLE_RE = re.compile(r"^\s*Статья\s+(\d+[\.\d]*)", re.MULTILINE)
_CLAUSE_RE = re.compile(r"^\s*(\d+\.\d+[\.\d]*)\s+", re.MULTILINE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

_ARTICLE_REF_RE = re.compile(r"ст(?:ать[июяе]|\.)\s*(\d+[\.\d]*)", re.IGNORECASE)
_CHAPTER_REF_RE = re.compile(r"глав(?:[ауы]|ой)\s*(\d+[\.\d]*)", re.IGNORECASE)
_LAW_NAME_RE = re.compile(
    r"(?:Федеральный закон|ГК РФ|НК РФ|КоАП|УК РФ|ТК РФ)\s*(?:от\s+[\d.]+\s*)?(?:№\s*\d+[\-\d]*)?",
    re.IGNORECASE,
)

_EFFECTIVE_PATTERNS = [
    (re.compile(r"вступает в силу\s+(?:со дня|с)\s+([^\n,.]+)", re.IGNORECASE), 0.9),
    (re.compile(r"вступлени[ея]\s+в\s+силу\s+([^\n,.]+)", re.IGNORECASE), 0.7),
    (re.compile(r"с\s+даты\s+([^\n,.]+)", re.IGNORECASE), 0.5),
]

_SIGNING_DATE_RE = re.compile(r"(\d{1,2}\s+\S+\s+\d{4}\s*г?\.?)")

_LEGAL_PROMPT_RULES = (
    "- Обязательно указывай номер статьи/пункта, если он есть в контексте "
    "(например: «Согласно ст. 15 ФЗ-XXX» или «п. 3.2 Договора»).\n"
    "- Не перефразируй формулировки нормативных актов — цитируй максимально близко к тексту.\n"
)


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
        min_articles = int(self._get("fingerprint_min_articles"))
        has_articles = len(_ARTICLE_RE.findall(text)) >= min_articles
        has_chapters = bool(_CHAPTER_RE.search(text))
        return has_articles and has_chapters

    def classify_score(self, text: str) -> float:
        score = 0.0
        score += 0.5 * len(_ARTICLE_RE.findall(text))
        score += 0.3 * len(_CHAPTER_RE.findall(text))
        score += 0.2 * len(_SECTION_RE.findall(text))
        if re.search(r"Федеральный закон|ГК РФ|НК РФ", text, re.IGNORECASE):
            score += 2.0
        if re.search(r"стороны\s+договорились|настоящ(им|его|ий)\s+(договор|соглашени)", text, re.IGNORECASE):
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
        refs = []
        for m in _ARTICLE_REF_RE.finditer(text):
            refs.append(ReferenceMatch("article", m.group(1)))
        for m in _CHAPTER_REF_RE.finditer(text):
            refs.append(ReferenceMatch("chapter", m.group(1)))
        for m in _LAW_NAME_RE.finditer(text):
            refs.append(ReferenceMatch("law_name", m.group(0).strip()))
        return refs

    def extract_effective_date(self, text: str) -> EffectiveDateCandidate | None:
        return extract_effective_date_generic(text, _EFFECTIVE_PATTERNS, _SIGNING_DATE_RE)

    def prompt_addendum(self, breadth: str, as_of_date: date | None = None) -> str | None:
        return _LEGAL_PROMPT_RULES + versioned_prompt_date_stamp(as_of_date)

    def retrieval_fallback_to_full_corpus(self) -> bool:
        return True
