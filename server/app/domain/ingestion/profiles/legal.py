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
from domain.ingestion.profiles.patterns import (
    PARAGRAPH_RE,
    POINT_RE,
    SENTENCE_RE,
    SUBPOINT_LETTER_RE,
    SUBPOINT_NUM_RE,
    regulatory_unit_label,
)
from domain.ingestion.profiles.types import (
    BoundaryLevel,
    ConfigDefault,
    ReferenceMatch,
)
from domain.shared.profile_policy import AnswerPolicy


if TYPE_CHECKING:
    from domain.ingestion.profiles.types import (
        EffectiveDateCandidate,
    )


# Official texts often set headings in capitals ("ГЛАВА 3"), so headings ignore case.
_CHAPTER_RE = re.compile(r"^\s*Глава\s+(\d+(?:\.\d+)*)", re.MULTILINE | re.IGNORECASE)
_SECTION_RE = re.compile(r"^\s*Раздел\s+(\d+(?:\.\d+)*)", re.MULTILINE | re.IGNORECASE)
_ARTICLE_RE = re.compile(r"^\s*Статья\s+(\d+(?:[.-]\d+)*)", re.MULTILINE | re.IGNORECASE)

_ARTICLE_REF_RE = re.compile(r"ст(?:ать[июяе]|\.)\s*(\d+(?:[.-]\d+)*)", re.IGNORECASE)
_CHAPTER_REF_RE = re.compile(r"глав(?:[ауы]|ой)\s*(\d+[\.\d]*)", re.IGNORECASE)
_LAW_NAME_RE = re.compile(
    r"(?:Федеральный закон|Федеральный конституционный закон|ГК "
    r"РФ|НК РФ|КоАП|УК РФ|ТК РФ)"
    r"\s*(?:от\s+[\d.]+\s*)?(?:№\s*[\d]+(?:-[\dА-ЯЁ]+)*)?",
    re.IGNORECASE,
)
_LAW_NUMBER_RE = re.compile(
    r"^\s*(?:Федеральный закон|Федеральный конституционный закон)"
    r"\s*(?:от\s+(?P<date>[^№N\n]+?)\s*)?(?:№|N)\s*(?P<number>[\d]+(?:-[\dА-ЯЁ]+)*)",
    re.IGNORECASE | re.MULTILINE,
)

_EFFECTIVE_PATTERNS = [
    (re.compile(r"вступает в силу\s+(?:со дня|с)\s+([^\n,]+)", re.IGNORECASE), 0.9),
    (re.compile(r"вступлени[ея]\s+в\s+силу\s+([^\n,]+)", re.IGNORECASE), 0.7),
    (re.compile(r"с\s+даты\s+([^\n,]+)", re.IGNORECASE), 0.5),
]

_SIGNING_DATE_RE = re.compile(r"(\d{1,2}\s+\S+\s+\d{4}\s*г?\.?)")


class LegalDomainProfile(SettingsBackedProfile):
    answer_policy = AnswerPolicy(
        "- Обязательно указывай номер статьи/пункта, если он есть в контексте (например: "
        "«Согласно ст. 15 ФЗ-XXX» или «п. 3.2 Договора»).\n- Не перефразируй формулировки "
        "нормативных актов — цитируй максимально близко к тексту.\n",
        fallback_to_full_corpus=True,
    )
    query_patterns: tuple[str, ...] = (
        "вправе\\s+ли",
        "обязан\\s+ли",
        "подлежит\\s+ли",
        "несёт\\s+ли\\s+ответственность",
        "стать[юяе]\\s+\\d+",
        "пункт[ае]?\\s+\\d+",
        "в\\s+соответствии\\s+с",
        "согласно\\s+(закону|договору|статье)",
        "нарушени[ея]\\s+(условий|закона)",
    )
    query_priority = 1
    key = "legal"
    display_name = "Нормативные акты"
    is_versioned = True

    def config_defaults(self) -> list[ConfigDefault]:
        return [
            ConfigDefault(f"{self.key}_chunk_target_tokens", "400", "int", "Target chunk tokens", 50, 8192),
            ConfigDefault(f"{self.key}_chunk_max_tokens", "800", "int", "Maximum chunk tokens", 50, 8192),
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

        if re.search(r"Федеральный закон|ГК РФ|НК РФ", text, re.IGNORECASE):
            score += 2.0

        if re.search(
            r"стороны\s+договорились|настоящ(им|его|ий)\s+(договор|соглашени)",
            text,
            re.IGNORECASE,
        ):
            score += 1.5

        return score

    def content_boundaries(self) -> list[BoundaryLevel]:
        return [
            BoundaryLevel("chapter", _CHAPTER_RE, packing_barrier=True, order_by_position=True),
            BoundaryLevel(
                "section", _SECTION_RE, always_split=True, packing_barrier=True, order_by_position=True
            ),
            BoundaryLevel("article", _ARTICLE_RE, always_split=True, packing_barrier=True),
            BoundaryLevel("point", POINT_RE),
            BoundaryLevel("subpoint_num", SUBPOINT_NUM_RE),
            BoundaryLevel("subpoint", SUBPOINT_LETTER_RE),
            BoundaryLevel("paragraph", PARAGRAPH_RE),
            BoundaryLevel("sentence", SENTENCE_RE),
        ]

    def unit_label(self, kind: str, value: str) -> str:
        return regulatory_unit_label(kind, value)

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
