"""Unversioned agreements, split by sections and nested clauses."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from domain.ingestion.profiles.base import SettingsBackedProfile
from domain.ingestion.profiles.patterns import PARAGRAPH_RE, SENTENCE_RE, SUBPOINT_LETTER_RE
from domain.ingestion.profiles.types import (
    BoundaryLevel,
    ConfigDefault,
    ReferenceMatch,
)
from domain.shared.profile_policy import AnswerPolicy


if TYPE_CHECKING:
    from domain.ingestion.profiles.types import EffectiveDateCandidate


CONTRACT_TITLE_RE = re.compile(r"^[ \t]*(?:ДОГОВОР|СОГЛАШЕНИЕ)\b", re.MULTILINE | re.IGNORECASE)
PARTIES_RE = re.compile(r"\bстороны\b|именуем\w*\s+в\s+дальнейшем", re.IGNORECASE)
SECTION_RE = re.compile(r"^[ \t]*(\d+)\.[ \t]+[А-ЯЁ]", re.MULTILINE)
CLAUSE_RE = re.compile(r"^[ \t]*(\d+\.\d+)\.?[ \t]+", re.MULTILINE)
SUBCLAUSE_RE = re.compile(r"^[ \t]*(\d+\.\d+\.\d+)\.?[ \t]+", re.MULTILINE)
APPENDIX_RE = re.compile(r"^[ \t]*Приложение\s*№?\s*(\d+)", re.MULTILINE | re.IGNORECASE)
CLAUSE_REF_RE = re.compile(r"\b(?:п\.|пункт[ае]?)\s*(\d+(?:\.\d+)+)", re.IGNORECASE)
SECTION_REF_RE = re.compile(r"\bраздел[ае]?\s+(\d+)", re.IGNORECASE)
APPENDIX_REF_RE = re.compile(r"\bПриложени[ея]\s*№?\s*(\d+)", re.IGNORECASE)
NUMBER_RE = re.compile(r"\b(?:договор|соглашение)\s*№\s*([\w/-]+)", re.IGNORECASE)
DATE_RE = re.compile(r"\b(?:\d{1,2}[./]\d{1,2}[./]\d{4}|\d{1,2}\s+[а-яё]+\s+\d{4}\s*г?\.?)", re.IGNORECASE)
ROLE_RE = re.compile(r"\b(Заказчик|Исполнитель|Покупатель|Поставщик|Арендатор|Арендодатель)\b", re.IGNORECASE)


class ContractDomainProfile(SettingsBackedProfile):
    key = "contract"
    display_name = "Договоры"
    is_versioned = False
    query_patterns: tuple[str, ...] = (r"\bдоговор\w*\b", r"\bсоглашени\w*\b")
    query_priority = 2
    # Retrieval is not narrowed to contracts: questions about agreements often
    # need the governing law ("договор ... по статье 619 ГК"), and a domain-first
    # search would drop legal acts whenever any contract chunk matches.
    answer_policy = AnswerPolicy(
        "- Ссылайся на конкретный пункт или раздел договора, если он есть в контексте.\n"
        "- Нормы закона цитируй с указанием статьи и акта.\n",
        fallback_to_full_corpus=False,
    )

    def config_defaults(self) -> list[ConfigDefault]:
        return [
            ConfigDefault("contract_chunk_target_tokens", "400", "int", "Target chunk tokens", 50, 8192),
            ConfigDefault("contract_chunk_max_tokens", "800", "int", "Maximum chunk tokens", 50, 8192),
            ConfigDefault("classification_threshold", "2.0", "float", "Classification threshold", 0, 20),
        ]

    def structural_fingerprint(self, text: str) -> bool:
        return bool(
            CONTRACT_TITLE_RE.search(text[:1500]) and PARTIES_RE.search(text) and SECTION_RE.search(text)
        )

    def classify_score(self, text: str) -> float:
        return 3.0 if self.structural_fingerprint(text) else 0.0

    def content_boundaries(self) -> list[BoundaryLevel]:
        return [
            BoundaryLevel("appendix", APPENDIX_RE, always_split=True, packing_barrier=True),
            BoundaryLevel("section", SECTION_RE, always_split=True, packing_barrier=True),
            BoundaryLevel("clause", CLAUSE_RE, always_split=True),
            BoundaryLevel("subclause", SUBCLAUSE_RE, always_split=True),
            BoundaryLevel("subpoint", SUBPOINT_LETTER_RE),
            BoundaryLevel("paragraph", PARAGRAPH_RE),
            BoundaryLevel("sentence", SENTENCE_RE),
        ]

    def unit_label(self, kind: str, value: str) -> str:
        label = {
            "appendix": "Приложение №",
            "section": "Раздел",
            "clause": "Пункт",
            "subclause": "Подпункт",
            "subpoint": "Подпункт",
        }.get(kind)
        return f"{label} {value}" if label else ""

    def extract_references(self, text: str) -> list[ReferenceMatch]:
        refs = [
            ReferenceMatch(kind, match.group(1))
            for kind, pattern in (
                ("clause", CLAUSE_REF_RE),
                ("section", SECTION_REF_RE),
                ("appendix", APPENDIX_REF_RE),
            )
            for match in pattern.finditer(text)
        ]
        header = text[:1500]

        if match := NUMBER_RE.search(header):
            refs.append(ReferenceMatch("contract_number", match.group(1)))

        if match := DATE_RE.search(header):
            refs.append(ReferenceMatch("contract_date", match.group()))

        refs.extend(
            ReferenceMatch("party_role", match.group(1).capitalize()) for match in ROLE_RE.finditer(text)
        )
        return refs

    def extract_effective_date(self, text: str) -> EffectiveDateCandidate | None:
        del text  # Agreements do not participate in act-version timelines.
        return None
